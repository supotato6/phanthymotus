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
import threading
import time

import cv2
from ctypes import POINTER, c_void_p, byref, c_size_t
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── MCP 工具元数据 ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "obstacle",
        "type": "processor",
        "multiInstance": False,
        "description": "障碍物距离检测：png=室内(DA2 metric INT8+ROI min+isotonic)，jpg=室外(yolo26n depth+seg -> 掩码 p5)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["info", "detect"]},
                "image_path": {"type": "string", "description": "输入图片路径，扩展名决定室内/室外"},
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

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        """x: [1,3,H,W] float32 归一化输入；返回输出 numpy 列表。"""
        x = np.ascontiguousarray(x)
        outs = [np.empty(tuple(self.engine.get_tensor_shape(nm)), np.float32) for nm in self.out_names]
        in_dev = c_void_p(); in_host = c_void_p()
        rc = self._rt.cudaMalloc(byref(in_dev), c_size_t(x.nbytes))
        if rc != 0:
            raise RuntimeError(f"cudaMalloc failed rc={rc}")
        rc = self._rt.cudaMallocHost(byref(in_host), c_size_t(x.nbytes))
        if rc != 0:
            self._rt.cudaFree(in_dev)
            raise RuntimeError(f"cudaMallocHost failed rc={rc}")
        ctypes.memmove(in_host, x.ctypes.data_as(c_void_p), x.nbytes)
        self._rt.cudaMemcpy(in_dev, in_host, c_size_t(x.nbytes), 2)
        devs, hosts = [in_dev], [in_host]
        try:
            for o in outs:
                d = c_void_p(); self._rt.cudaMalloc(byref(d), c_size_t(o.nbytes))
                h = c_void_p(); self._rt.cudaMallocHost(byref(h), c_size_t(o.nbytes))
                devs.append(d); hosts.append(h)
            self.ctx.set_tensor_address(self.in_name, int(in_dev.value))
            for i, nm in enumerate(self.out_names):
                self.ctx.set_tensor_address(nm, int(devs[i + 1].value))
            ok = self.ctx.execute_v2([int(d.value) for d in devs])
            if not ok:
                raise RuntimeError("engine execute failed")
            for i, (o, d, h) in enumerate(zip(outs, devs[1:], hosts[1:])):
                self._rt.cudaMemcpy(h, d, c_size_t(o.nbytes), 1)
                ctypes.memmove(o.ctypes.data_as(c_void_p), h, o.nbytes)
        finally:
            for d, h in zip(devs, hosts):
                self._rt.cudaFree(d)
                self._rt.cudaFreeHost(h)
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


class ObstaclePlugin:
    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = plugin_cfg
        self._model_dir = plugin_cfg.get("model_dir", "/models/obstacle")
        self._lock = threading.Lock()
        self._indoor_eng: Optional[_TrtEngine] = None
        self._out_depth_eng: Optional[_TrtEngine] = None
        self._out_seg_eng: Optional[_TrtEngine] = None
        self._indoor_knots = None
        self._load_error = None
        self._load_status = "pending"
        try:
            self._load_models()
        except Exception as e:
            self._load_error = str(e)
            self._load_status = "error"
            log.error(f"[obstacle] model load failed: {e}", exc_info=True)

    # ── 模型加载 ──────────────────────────────────────────────────────────
    def _load_models(self):
        ind = self._cfg.get("indoor", {})
        out = self._cfg.get("outdoor", {})
        # 室内：DA2 metric INT8 + isotonic
        eng_path = os.path.join(self._model_dir, ind.get("engine", "depth_anything_v2_metric_hypersim_vits_int8.trt"))
        calib_path = os.path.join(self._model_dir, ind.get("calib", "calib_isotonic_d_roi_min.json"))
        self._indoor_eng = _TrtEngine(eng_path)
        with open(calib_path) as f:
            cal = json.load(f)
        self._indoor_knots = (np.asarray(cal["x_knots"], np.float64), np.asarray(cal["y_knots"], np.float64))
        # 室外：depth + seg
        self._out_depth_eng = _TrtEngine(os.path.join(self._model_dir, out.get("depth_engine", "yolo26n-depth_int8.trt")))
        self._out_seg_eng = _TrtEngine(os.path.join(self._model_dir, out.get("seg_engine", "yolo26n-seg_fp16.trt")))
        self._load_status = "ready"
        log.info(f"[obstacle] models ready: indoor={os.path.basename(eng_path)} "
                 f"outdoor={out.get('depth_engine')} + {out.get('seg_engine')}")

    def get_tools(self) -> list:
        return TOOLS

    # ── MCP 分发 ──────────────────────────────────────────────────────────
    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == self.PREFIX else name
        if action == "info":
            return {
                "name": "Obstacle",
                "state": self._load_status,
                "error": self._load_error,
                "indoor": "DA2-Small metric-hypersim INT8 + ROI min + isotonic",
                "outdoor": "yolo26n-depth INT8 + yolo26n-seg -> mask p5 + scale/bias",
                "dispatch_rule": "png=indoor, jpg=outdoor",
                "model_dir": self._model_dir,
            }
        if action == "detect":
            return self._detect(args)
        return None

    # ── 检测入口 ──────────────────────────────────────────────────────────
    def _detect(self, args: dict) -> dict:
        image_path = (args.get("image_path") or "").strip()
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
        return {
            "ok": True,
            "mode": mode,
            "distance_m": round(float(dist), 3),
            "fallback": bool(info.get("fallback", False)),
            "image_path": image_path,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }

    # ── 室内：DA2 metric INT8 + ROI min + isotonic ────────────────────────
    def _detect_indoor(self, bgr) -> tuple[float, dict]:
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
        return pred, {"fallback": False}

    # ── 室外：yolo26n depth + seg ─────────────────────────────────────────
    def _detect_outdoor(self, bgr) -> tuple[float, dict]:
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
