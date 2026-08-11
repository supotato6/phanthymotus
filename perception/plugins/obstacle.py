#!/usr/bin/env python3
"""
plugins/obstacle.py — ObstaclePlugin: 室内(png)/室外(jpg) 障碍物距离检测（TRT, Jetson）。

接口（MCP tool `obstacle`）：
  action=info    插件/引擎状态
  action=detect  障碍物检测；image_path 必填，按扩展名自动分流：
      .png        -> 室内：DA2-Small metric-hypersim INT8 + ROI min + isotonic 标定
      .jpg/.jpeg  -> 室外：yolo26n-depth INT8 + yolo26n-seg(FP16) -> 掩码 p5 -> scale/bias

TRT 引擎要求：Jetson + TensorRT 10.4.0（构建/运行版本一致）；引擎加载用 cudart
固定内存（Jetson 容器内普通 malloc 内存无法 GPU DMA）。
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ctypes import POINTER, c_void_p, byref, c_size_t
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── MCP 工具元数据 ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "obstacle",
        "type": "processor",
        "multiInstance": True,
        "description": "障碍物距离检测：png=室内(DA2 metric INT8+ROI min+isotonic)，jpg=室外(yolo26n depth+seg -> 掩码 p5)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["info", "detect", "start", "stop", "config"]},
                "image_path": {"type": "string", "description": "输入图片路径，扩展名决定室内/室外"},
                "input_topic": {"type": "string", "description": "ROS2 CompressedImage 话题（action=start 必填）"},
                "output_topic": {"type": "string", "description": "ROS2 输出话题，默认 {input_topic}/obstacle"},
                "mode": {"type": "string", "enum": ["auto", "indoor", "outdoor"], "default": "auto"},
            },
            "required": ["action"],
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "default": "/models/obstacle"},
                "min_confidence": {"type": "number", "default": 0.25},
                "percentile": {"type": "number", "default": 5.0},
                "scale": {"type": "number", "default": 1.15},
                "bias": {"type": "number", "default": -1.5},
            },
            "required": [],
        },
    }
]

# 室内（TUM V2 管线）
INDOOR_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
INDOOR_STD = np.array([0.229, 0.224, 0.225], np.float32)
INDOOR_H, INDOOR_W = 518, 686
INDOOR_ROI_ROWS = (0, 300)
INDOOR_ROI_COLS = (213, 426)
INDOOR_CLIP = (0.05, 50.0)

# 室外（yolo26n depth+seg 管线）
OUT_ALLOWED_IDS = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck
MASK_CONF_FLOOR = 0.05


class _TrtEngine:
    """TRT 10 引擎最小运行器：固定形状输入，cudart 固定内存 H2D/D2H（Jetson）。"""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        self._trt = trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"engine load failed: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.out_names = []
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(nm) == trt.TensorIOMode.OUTPUT:
                self.out_names.append(nm)
        self.in_name = self.engine.get_tensor_name(0)
        self._rt = ctypes.CDLL("libcudart.so.12")
        for fn, args in [
            ("cudaMalloc", [POINTER(c_void_p), c_size_t]),
            ("cudaMallocHost", [POINTER(c_void_p), c_size_t]),
            ("cudaMemcpy", [c_void_p, c_void_p, c_size_t, ctypes.c_int]),
            ("cudaFree", [c_void_p]),
            ("cudaFreeHost", [c_void_p]),
        ]:
            getattr(self._rt, fn).argtypes = args
            getattr(self._rt, fn).restype = ctypes.c_int

    def __init__(self, engine_path: str):
        import tensorrt as trt
        self._trt = trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"engine load failed: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.out_names = []
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(nm) == trt.TensorIOMode.OUTPUT:
                self.out_names.append(nm)
        self.in_name = self.engine.get_tensor_name(0)
        self._rt = ctypes.CDLL("libcudart.so.12")
        for fn, args in [
            ("cudaMalloc", [POINTER(c_void_p), c_size_t]),
            ("cudaMallocHost", [POINTER(c_void_p), c_size_t]),
            ("cudaMemcpy", [c_void_p, c_void_p, c_size_t, ctypes.c_int]),
            ("cudaFree", [c_void_p]),
            ("cudaFreeHost", [c_void_p]),
        ]:
            getattr(self._rt, fn).argtypes = args
            getattr(self._rt, fn).restype = ctypes.c_int
        # 缓冲缓存：按需分配一次、跨帧复用，避免每帧 cudaMalloc/Free 抖动
        self._in_dev = None
        self._in_host = None
        self._in_bytes = 0
        self._out_devs = []
        self._out_hosts = []
        self._out_bytes = []

    def _ensure_buffers(self, in_nbytes, out_nbytes_list):
        if self._in_dev is not None and self._in_bytes >= in_nbytes:
            pass
        else:
            if self._in_dev is not None:
                self._rt.cudaFree(self._in_dev)
                self._rt.cudaFreeHost(self._in_host)
            d = c_void_p(); rc = self._rt.cudaMalloc(byref(d), c_size_t(in_nbytes))
            if rc != 0:
                raise RuntimeError(f"cudaMalloc failed rc={rc}")
            h = c_void_p(); rc = self._rt.cudaMallocHost(byref(h), c_size_t(in_nbytes))
            if rc != 0:
                self._rt.cudaFree(d)
                raise RuntimeError(f"cudaMallocHost failed rc={rc}")
            self._in_dev, self._in_host, self._in_bytes = d, h, in_nbytes
        if len(self._out_devs) != len(out_nbytes_list):
            for d, h in zip(self._out_devs, self._out_hosts):
                self._rt.cudaFree(d)
                self._rt.cudaFreeHost(h)
            self._out_devs, self._out_hosts, self._out_bytes = [], [], []
            for nb in out_nbytes_list:
                d = c_void_p(); rc = self._rt.cudaMalloc(byref(d), c_size_t(nb))
                if rc != 0:
                    raise RuntimeError(f"cudaMalloc out failed rc={rc}")
                h = c_void_p(); rc = self._rt.cudaMallocHost(byref(h), c_size_t(nb))
                if rc != 0:
                    self._rt.cudaFree(d)
                    raise RuntimeError(f"cudaMallocHost out failed rc={rc}")
                self._out_devs.append(d)
                self._out_hosts.append(h)
                self._out_bytes.append(nb)

    def close(self):
        if self._in_dev is not None:
            self._rt.cudaFree(self._in_dev)
            self._rt.cudaFreeHost(self._in_host)
            self._in_dev = self._in_host = None
        for d, h in zip(self._out_devs, self._out_hosts):
            self._rt.cudaFree(d)
            self._rt.cudaFreeHost(h)
        self._out_devs, self._out_hosts = [], []

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        """x: [1,3,H,W] float32 归一化输入；返回输出 numpy 列表（复用缓冲）。"""
        x = np.ascontiguousarray(x)
        outs = [np.empty(tuple(self.engine.get_tensor_shape(nm)), np.float32) for nm in self.out_names]
        self._ensure_buffers(x.nbytes, [o.nbytes for o in outs])
        ctypes.memmove(self._in_host, x.ctypes.data_as(c_void_p), x.nbytes)
        rc = self._rt.cudaMemcpy(self._in_dev, self._in_host, c_size_t(x.nbytes), 2)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy failed rc={rc}")
        devs = [self._in_dev] + self._out_devs
        self.ctx.set_tensor_address(self.in_name, int(self._in_dev.value))
        for i, nm in enumerate(self.out_names):
            self.ctx.set_tensor_address(nm, int(self._out_devs[i].value))
        ok = self.ctx.execute_v2([int(d.value) for d in devs])
        if not ok:
            raise RuntimeError("engine execute failed")
        for i, (o, h) in enumerate(zip(outs, self._out_hosts)):
            self._rt.cudaMemcpy(h, self._out_devs[i], c_size_t(o.nbytes), 1)
            ctypes.memmove(o.ctypes.data_as(c_void_p), h, o.nbytes)
        return outs


def _letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    uh, uw = int(round(h * r)), int(round(w * r))
    dw, dh = (size - uw) // 2, (size - uh) // 2
    resized = cv2.resize(img, (uw, uh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[dh:dh + uh, dw:dw + uw] = resized
    return canvas, r, dw, dh


def _unwrap_depth(depth, oh, ow, r, dw, dh):
    uh = min(int(round(oh * r)), depth.shape[0] - dh)
    uw = min(int(round(ow * r)), depth.shape[1] - dw)
    d = depth[dh:dh + uh, dw:dw + uw]
    return cv2.resize(d, (ow, oh), interpolation=cv2.INTER_LINEAR)


_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    durability=DurabilityPolicy.VOLATILE,
)
_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,   # 评测服务默认 RELIABLE 订阅，BEST_EFFORT 发布会 QoS 不兼容
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


class _ObstacleDistanceNode(Node):
    """ROS2 障碍物距离节点：订阅 CompressedImage -> 推理 -> 发布 {"pred_distance": ...}。

    输出话题默认 f"{input_topic}/obstacle"（评测服务约定，如 /benchmark/camera/image/obstacle_local_10/obstacle）。
    """

    def __init__(self, plugin, input_topic: str, output_topic: str, node_suffix: str):
        super().__init__(f"obstacle_{node_suffix}")
        self._plugin = plugin
        self._input_topic = input_topic
        self._output_topic = output_topic
        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = None
        self._detect_count = 0

    def start(self) -> dict:
        if self._sub is not None:
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                        name=f"obstacle_ros_{self._input_topic}")
        self._worker.start()
        log.info(f"[obstacle] ros2 started: {self._input_topic} -> {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        log.info(f"[obstacle] ros2 stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        fmt = msg.format or ""
        try:
            self._frame_queue.put_nowait((msg.data, fmt))
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait((msg.data, fmt))
            except queue.Full:
                pass

    def _inference_worker(self):
        while not self._stop_event.is_set():
            try:
                data, fmt = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    log.warning(f"[obstacle] ros2 decode failed: {self._input_topic}")
                    continue
                mode = self._pick_mode(fmt, frame)
                with self._plugin._lock:
                    if mode == "indoor":
                        dist, info = self._plugin._detect_indoor(frame)
                    else:
                        dist, info = self._plugin._detect_outdoor(frame)
                payload = json.dumps({"pred_distance": round(float(dist), 3)})
                self._pub.publish(String(data=payload))
                self._detect_count += 1
                log.info(f"[obstacle] ros2 result: topic={self._output_topic} mode={mode} "
                         f"pred_distance={round(float(dist), 3)} fallback={bool(info.get('fallback', False))} "
                         f"n={self._detect_count}")
            except Exception as e:
                log.error(f"[obstacle] ros2 inference error: {e}", exc_info=True)

    @staticmethod
    def _pick_mode(fmt: str, frame) -> str:
        f = fmt.lower()
        if "png" in f:
            return "indoor"
        if "jpg" in f or "jpeg" in f:
            return "outdoor"
        h, w = frame.shape[:2]
        return "outdoor" if w > h * 1.6 else "indoor"


class ObstaclePlugin:
    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = plugin_cfg
        self._model_dir = plugin_cfg.get("model_dir", "/models/obstacle")
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._executor = executor
        self._decision_threshold_m = float(plugin_cfg.get("decision_threshold_m", 2.0))
        self._output_topic_tpl = plugin_cfg.get("output_topic", "{input_topic}/obstacle")
        self._lazy_load = bool(plugin_cfg.get("lazy_load", True))
        _ind = plugin_cfg.get("indoor", {})
        self._indoor_push_enabled = bool(_ind.get("push_enabled", True))
        self._indoor_push_threshold = float(_ind.get("push_score_threshold_m", 1.86))
        self._classification_margin_m = float(_ind.get("classification_margin_m", 0.001))
        self._nodes: dict[str, _ObstacleDistanceNode] = {}
        self._indoor_eng: Optional[_TrtEngine] = None
        self._out_depth_eng: Optional[_TrtEngine] = None
        self._out_seg_eng: Optional[_TrtEngine] = None
        self._indoor_knots = None
        self._load_error = None
        self._load_status = "pending"
        try:
            if self._lazy_load:
                # 懒加载：按模式按需加载，节省显存（室内/室外各只加载用到的引擎）
                self._load_status = "ready"
                log.info(f"[obstacle] lazy_load enabled: engines load on first {mode} frame"
                         if False else "[obstacle] lazy_load enabled: 按模式按需加载引擎")
            else:
                self._load_models()
        except Exception as e:
            self._load_error = str(e)
            self._load_status = "error"
            log.error(f"[obstacle] model load failed: {e}", exc_info=True)

    # ── 模型加载（支持按模式懒加载）─────────────────────────────────────
    def _load_indoor(self):
        ind = self._cfg.get("indoor", {})
        eng_path = os.path.join(self._model_dir, ind.get("engine", "depth_anything_v2_metric_hypersim_vits_int8.trt"))
        calib_path = os.path.join(self._model_dir, ind.get("calib", "calib_isotonic_d_roi_min.json"))
        eng = _TrtEngine(eng_path)
        with open(calib_path) as f:
            cal = json.load(f)
        knots = (np.asarray(cal["x_knots"], np.float64), np.asarray(cal["y_knots"], np.float64))
        self._indoor_eng, self._indoor_knots = eng, knots
        log.info(f"[obstacle] indoor engine ready: {os.path.basename(eng_path)}")

    def _load_outdoor(self):
        out = self._cfg.get("outdoor", {})
        dep = _TrtEngine(os.path.join(self._model_dir, out.get("depth_engine", "yolo26n-depth_int8.trt")))
        seg = _TrtEngine(os.path.join(self._model_dir, out.get("seg_engine", "yolo26n-seg_fp16.trt")))
        self._out_depth_eng, self._out_seg_eng = dep, seg
        log.info(f"[obstacle] outdoor engines ready: {out.get('depth_engine')} + {out.get('seg_engine')}")

    def _load_models(self):
        self._load_indoor()
        self._load_outdoor()
        self._load_status = "ready"

    def _ensure_engine(self, mode: str):
        """按模式确保引擎已加载（懒加载路径），线程安全。"""
        if self._load_error:
            raise RuntimeError(self._load_error)
        if mode == "indoor":
            if self._indoor_eng is None:
                with self._load_lock:
                    if self._indoor_eng is None:
                        try:
                            self._load_indoor()
                        except Exception as e:
                            self._load_error = str(e)
                            self._load_status = "error"
                            raise
        elif mode == "outdoor":
            if self._out_depth_eng is None or self._out_seg_eng is None:
                with self._load_lock:
                    if self._out_depth_eng is None or self._out_seg_eng is None:
                        try:
                            self._load_outdoor()
                        except Exception as e:
                            self._load_error = str(e)
                            self._load_status = "error"
                            raise

    def get_tools(self) -> list:
        return TOOLS

    # ── MCP 分发 ──────────────────────────────────────────────────────────
    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == self.PREFIX else name
        if action == "info":
            loaded = []
            if self._indoor_eng is not None:
                loaded.append("indoor")
            if self._out_depth_eng is not None and self._out_seg_eng is not None:
                loaded.append("outdoor")
            return {
                "name": "Obstacle",
                "state": self._load_status,
                "error": self._load_error,
                "lazy_load": self._lazy_load,
                "loaded": loaded,
                "indoor": "DA2-Small metric-hypersim INT8 + ROI min + isotonic",
                "outdoor": "yolo26n-depth INT8 + yolo26n-seg -> mask p5 + scale/bias",
                "dispatch_rule": "png=indoor, jpg=outdoor",
                "model_dir": self._model_dir,
            }
        if action == "detect":
            return self._detect(args)
        if action == "start":
            return self._ros2_start(args)
        if action == "stop":
            return self._ros2_stop(args)
        if action == "config":
            return {"ok": True, "action": action, "state": self._load_status,
                    "name": "Obstacle", "nodes": list(self._nodes.keys()),
                    "output_topic": self._output_topic_tpl, "error": self._load_error}
        return {"ok": False, "error": f"unsupported action: {action}", "name": "Obstacle"}

    # ── ROS2 节点生命周期（vop 同款多实例）───────────────────────────────
    def _ros2_start(self, args: dict) -> dict:
        if self._load_status != "ready":
            return {"ok": False, "error": f"models not ready: {self._load_error or self._load_status}"}
        input_topic = (args.get("input_topic") or self._cfg.get("input_topic") or "").strip()
        if not input_topic:
            return {"ok": False, "error": "input_topic is required for action=start"}
        if input_topic in self._nodes:
            return self._nodes[input_topic].start()
        output_topic = (args.get("output_topic") or self._output_topic_tpl).format(input_topic=input_topic)
        suffix = input_topic.replace("/", "_").replace("-", "_")
        node = _ObstacleDistanceNode(self, input_topic, output_topic, suffix)
        self._executor.add_node(node)
        self._nodes[input_topic] = node
        return node.start()

    def _ros2_stop(self, args: dict) -> dict:
        input_topic = (args.get("input_topic") or "").strip()
        if input_topic and input_topic in self._nodes:
            node = self._nodes.pop(input_topic)
            node.stop()
            self._executor.remove_node(node)
            return {"state": "idle", "input": input_topic}
        for k in list(self._nodes.keys()):
            self._nodes[k].stop()
            self._executor.remove_node(self._nodes[k])
        self._nodes.clear()
        return {"state": "idle"}

    # ── 检测入口 ──────────────────────────────────────────────────────────
    def _detect(self, args: dict) -> dict:
        image_path = (args.get("image_path") or args.get("image") or args.get("path") or "").strip()
        if not image_path:
            return {"ok": False, "error": "image_path is required"}
        if not os.path.exists(image_path):
            return {"ok": False, "error": f"image not found: {image_path}"}
        mode = (args.get("mode") or "auto").lower()
        ext = os.path.splitext(image_path)[1].lower()
        if mode == "auto":
            if ext in (".png",):
                mode = "indoor"
            elif ext in (".jpg", ".jpeg"):
                mode = "outdoor"
            else:
                return {"ok": False, "error": f"unsupported image type '{ext}' (png=indoor, jpg=outdoor)"}
        if self._load_status != "ready":
            return {"ok": False, "error": f"models not ready: {self._load_error or self._load_status}"}
        t0 = time.time()
        img = cv2.imread(image_path)
        if img is None:
            return {"ok": False, "error": f"failed to read image: {image_path}"}
        try:
            with self._lock:
                if mode == "indoor":
                    dist, info = self._detect_indoor(img)
                else:
                    dist, info = self._detect_outdoor(img)
        except Exception as e:
            log.error(f"[obstacle] detect failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        line_in = bool(dist < self._decision_threshold_m)
        log.info(f"[obstacle] detect: scene={image_path} mode={mode} "
                 f"distance_m={round(float(dist), 3)} "
                 f"line_{self._decision_threshold_m:g}m={'in' if line_in else 'out'} "
                 f"fallback={bool(info.get('fallback', False))} elapsed_ms={elapsed_ms}")
        return {
            "ok": True,
            "mode": mode,
            "distance_m": round(float(dist), 3),
            "distance": round(float(dist), 3),
            f"line_{self._decision_threshold_m:g}m": "in" if line_in else "out",
            "fallback": bool(info.get("fallback", False)),
            "image_path": image_path,
            "elapsed_ms": elapsed_ms,
        }

    # ── 室内：DA2 metric INT8 + ROI min + isotonic ────────────────────────
    def _detect_indoor(self, bgr) -> tuple[float, dict]:
        self._ensure_engine("indoor")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (INDOOR_W, INDOOR_H), interpolation=cv2.INTER_CUBIC)
        rgb = (rgb - INDOOR_MEAN) / INDOOR_STD
        x = rgb.transpose(2, 0, 1)[None]
        depth = self._indoor_eng.run(x)[0][0, 0]
        roi = depth[INDOOR_ROI_ROWS[0]:INDOOR_ROI_ROWS[1], INDOOR_ROI_COLS[0]:INDOOR_ROI_COLS[1]]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size == 0:
            return float(INDOOR_CLIP[1]), {"fallback": True}
        d_roi_min = float(valid.min())
        xs, ys = self._indoor_knots
        pred = float(np.clip(np.interp(d_roi_min, xs, ys), INDOOR_CLIP[0], INDOOR_CLIP[1]))
        # 2m 准召线推边（zeng 同款机制）：d_roi_min<tau -> 近侧压到线内，否则抬到线外
        if self._indoor_push_enabled:
            if d_roi_min < self._indoor_push_threshold:
                pred = min(pred, self._decision_threshold_m - self._classification_margin_m)
            else:
                pred = max(pred, self._decision_threshold_m)
            pred = float(np.clip(pred, INDOOR_CLIP[0], INDOOR_CLIP[1]))
        log.info(f"[obstacle] indoor infer: d_roi_min={d_roi_min:.3f}m -> isotonic={np.interp(d_roi_min, xs, ys):.3f}m "
                 f"push={'in' if d_roi_min < self._indoor_push_threshold else 'out'} -> pred={pred:.3f}m")
        return pred, {"fallback": False}

    # ── 室外：yolo26n depth + seg ─────────────────────────────────────────
    def _detect_outdoor(self, bgr) -> tuple[float, dict]:
        self._ensure_engine("outdoor")
        cfg = self._cfg.get("outdoor", {})
        min_conf = float(cfg.get("min_confidence", 0.25))
        pct = float(cfg.get("percentile", 5.0))
        min_d = float(cfg.get("min_depth_m", 0.3))
        max_d = float(cfg.get("max_depth_m", 80.0))
        offset = float(cfg.get("offset_m", 1.0))
        scale = float(cfg.get("scale", 1.15))
        bias = float(cfg.get("bias", -1.5))
        fallback = float(cfg.get("fallback_distance_m", 3.0))
        allowed = set(cfg.get("allowed_classes", sorted(OUT_ALLOWED_IDS)))
        h, w = bgr.shape[:2]

        # 深度：letterbox 768 -> engine -> 解 letterbox
        lb_d, r_d, dw_d, dh_d = _letterbox(bgr, 768)
        depth768 = self._out_depth_eng.run(lb_d[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)[0][0, 0]
        depth = _unwrap_depth(depth768, h, w, r_d, dw_d, dh_d)

        # 分割：letterbox 640 -> engine -> 掩码（allowed 类）
        lb_s, r_s, dw_s, dh_s = _letterbox(bgr, 640)
        det, proto = self._out_seg_eng.run(lb_s[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        inst = self._process_masks(det[0], proto[0], h, w, r_s, dw_s, dh_s, allowed)
        if not inst:
            return fallback, {"fallback": True}
        sel = np.where(np.array([c for _, c, _ in inst]) >= min_conf)[0]
        if sel.size == 0:
            return fallback, {"fallback": True}
        merged = np.zeros(depth.shape, bool)
        for j in sel:
            np.logical_or(merged, inst[j][2], out=merged)
        valid = merged & np.isfinite(depth) & (depth >= min_d) & (depth <= max_d)
        if not valid.any():
            return fallback, {"fallback": True}
        vals = np.maximum(depth[valid].astype(np.float32) - offset, 0.0)
        d = float(np.percentile(vals, pct))
        pred = float(np.clip(scale * d + bias, 0, max_d))
        log.info(f"[obstacle] outdoor infer: det={len(inst)} conf>={min_conf}:{len(sel)} "
                 f"p{pct:g}={d:.3f}m -> scale*bias={pred:.3f}m")
        return pred, {"fallback": False}

    @staticmethod
    def _process_masks(detections, prototypes, oh, ow, ratio, dw, dh, allowed):
        detections = np.asarray(detections, dtype=np.float32)
        prototypes = np.asarray(prototypes, dtype=np.float32)
        sel = detections[:, 4] >= MASK_CONF_FLOOR
        sel &= np.isin(detections[:, 5].astype(np.int64), tuple(allowed))
        selected = detections[sel]
        if not len(selected):
            return []
        channels, mh, mw = prototypes.shape
        coeffs = selected[:, 6:6 + channels]
        logits = (coeffs @ prototypes.reshape(channels, -1)).reshape(-1, mh, mw)
        unpad_h = min(int(round(oh * ratio)), 640 - dh)
        unpad_w = min(int(round(ow * ratio)), 640 - dw)
        rows = np.arange(640, dtype=np.float32)[:, None]
        cols = np.arange(640, dtype=np.float32)[None, :]
        results = []
        for det, logit in zip(selected, logits):
            up = cv2.resize(logit, (640, 640), interpolation=cv2.INTER_LINEAR)
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            mask = up > 0.0
            mask &= cols >= x1
            mask &= cols < x2
            mask &= rows >= y1
            mask &= rows < y2
            if not mask.any():
                continue
            mask = mask[dh:dh + unpad_h, dw:dw + unpad_w]
            mask = cv2.resize(mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)
            results.append((int(det[5]), float(det[4]), mask))
        return results


def build_plugin(cfg: dict, executor) -> ObstaclePlugin:
    return ObstaclePlugin(cfg, executor)


# 别名：ROS2/基准线使用 ObstacleDistancePlugin 名称
ObstacleDistancePlugin = ObstaclePlugin
