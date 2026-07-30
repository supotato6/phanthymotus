#!/usr/bin/env python3
"""
plugins/tts.py — TTSPlugin: VITS2-Mix INT8 PyTorch TTS.

Chinese-English mixed TTS using VITS2-Mix with INT8 quantized weights.
Model: G_B_final_int8.pth (35.4MB, trained on 柒小白 + mixed CN-EN data).
"""

from __future__ import annotations

import ctypes, gc, json, logging, os, queue, struct, sys, threading, time, types
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "tts",
        "type": "processor",
        "multiInstance": True,
        "description": "VITS2 INT8 TTS — speech synthesis with Chinese-English mixed support",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "speak", "info", "config"]},
                "input_topic": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "speaker_id": {"type": "integer", "default": 0, "scope": "shared"},
                "speed":      {"type": "number",  "default": 1.0, "scope": "shared"},
            },
            "required": []
        },
        "topic_in":  [{"format": "data/json",     "desc": "text to synthesize"}],
        "topic_out": [{"format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
    }
]


# ── Pure Python MAS (falls back from monotonic_align C extension) ──
def _maximum_path(value, mask, max_neg_val=-np.inf):
    dtype = value.dtype
    value = value.astype(np.float64)
    mask = mask.astype(np.float64)
    B, T_x, T_y = value.shape
    Q = np.full((B, T_x, T_y), max_neg_val, dtype=np.float64)
    Q[:, 0, 0] = value[:, 0, 0]
    for t in range(1, T_y):
        Q[:, 0, t] = Q[:, 0, t-1] + value[:, 0, t] if mask[:, 0, t] else max_neg_val
    for t_x in range(1, T_x):
        Q[:, t_x, 0] = value[:, t_x, 0] + max(Q[:, t_x-1, 0], max_neg_val if not mask[:, t_x, 0] else Q[:, t_x-1, 0])
    for t_x in range(1, T_x):
        for t_y in range(1, T_y):
            if mask[:, t_x, t_y]:
                Q[:, t_x, t_y] = value[:, t_x, t_y] + max(Q[:, t_x-1, t_y-1], Q[:, t_x-1, t_y])
    path = np.zeros((B, T_x, T_y), dtype=np.float64)
    path[:, T_x-1, T_y-1] = 1.0
    for t_x in range(T_x-1, -1, -1):
        for t_y in range(T_y-1, -1, -1):
            if t_x == 0 and t_y == 0: continue
            if t_x == 0: path[:, t_x, t_y-1] = 1.0
            elif t_y == 0: path[:, t_x-1, t_y] = 1.0
            else:
                best = np.argmax(np.array([Q[:, t_x-1, t_y-1], Q[:, t_x-1, t_y]]), axis=0)
                for b in range(B):
                    path[b, t_x-1, t_y-(1 if best[b]==0 else 0)] = 1.0
    return path.astype(dtype)


if "monotonic_align" not in sys.modules:
    _ma = types.ModuleType("monotonic_align")
    _ma.maximum_path = _maximum_path
    sys.modules["monotonic_align"] = _ma


# ── TTS Adapter ──────────────────────────────────────────────────────────────

class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...
    def synthesize_stream(self, text: str):
        yield self.synthesize(text)


# ── TRT TTS Adapter ────────────────────────────────────────────────────────

class TRTTSAdapter(TTSAdapter):
    """VITS2-Mix TensorRT TTS adapter — no PyTorch dependency.

    Uses ONNX Runtime for encoder, TRT engines for flow + decoder,
    and NumPy for iSTFT.  Memory ~280MB vs ~850MB for PyTorch.
    """

    def __init__(self, model_dir: str, trt_dir: str,
                 speaker_id: int = 0, speed: float = 1.0):
        self._speed = speed
        self._trt_dir = trt_dir

        # ── Frontend (same as PyTorch adapter) ──
        sys.path.insert(0, model_dir)
        from frontend.cleaner import clean_text_mix
        from frontend import cleaned_text_to_sequence_mix
        self._clean_text = clean_text_mix
        self._seq_mix = cleaned_text_to_sequence_mix

        # Read config directly — avoid importing vits2 (which requires torch)
        import json as _json
        class _HParams:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, _HParams(**v) if isinstance(v, dict) else v)
        with open(os.path.join(model_dir, "config.json"), "r") as _f:
            hps = _HParams(**_json.load(_f))

        # ── ONNX Runtime encoder ──
        import onnxruntime as ort
        encoder_path = os.path.join(trt_dir, "encoder_duration.onnx")
        self._encoder = ort.InferenceSession(encoder_path,
                                              providers=["CPUExecutionProvider"])

        # ── TRT engines (auto-build at first startup, GPU available at runtime) ─
        import tensorrt as trt
        import subprocess
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        flow_path = os.path.join(trt_dir, "flow.trt")
        dec_path = os.path.join(trt_dir, "decoder.trt")
        flow_onnx = os.path.join(self._trt_dir, "flow.onnx")
        dec_onnx = os.path.join(self._trt_dir, "decoder_spec.onnx")

        with open(flow_path, "rb") as f:
            self._flow_eng = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        with open(dec_path, "rb") as f:
            self._dec_eng = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        if self._flow_eng is None or self._dec_eng is None:
            raise RuntimeError(f"Failed to load TRT engines from {trt_dir}")

        # ── CUDA allocator ──
        import ctypes
        self._cuda = ctypes.CDLL("libcudart.so")
        self._cuda.cudaMalloc.restype = int
        self._cuda.cudaMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._cuda.cudaFree.restype = int
        self._cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self._cuda.cudaMemcpy.restype = int
        self._cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_size_t, ctypes.c_int]

        log.info(f"[tts] TRT adapter loaded: encoder={encoder_path}")

    def _gpu_alloc(self, size):
        ptr = ctypes.c_void_p(0)
        self._cuda.cudaMalloc(ctypes.byref(ptr), size)
        return ptr

    def _trt_run(self, engine, inputs, output_names):
        import tensorrt as trt
        ctx = engine.create_execution_context()
        gpu_ptrs = {}
        outputs = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                data = inputs[name].astype(np.float32)
                ctx.set_input_shape(name, data.shape)
                ptr = self._gpu_alloc(data.nbytes)
                self._cuda.cudaMemcpy(ptr, data.ctypes.data, data.nbytes, 1)  # H2D
                gpu_ptrs[name] = ptr
            else:
                shape = tuple(ctx.get_tensor_shape(name))
                outputs[name] = np.empty(shape, dtype=np.float32)
                ptr = self._gpu_alloc(outputs[name].nbytes)
                gpu_ptrs[name] = ptr

        bindings = [gpu_ptrs[engine.get_tensor_name(i)].value
                     for i in range(engine.num_io_tensors)]
        ctx.execute_v2(bindings)

        for name, arr in outputs.items():
            self._cuda.cudaMemcpy(arr.ctypes.data, gpu_ptrs[name], arr.nbytes, 2)  # D2H
        for p in gpu_ptrs.values():
            self._cuda.cudaFree(p)

        return tuple(outputs[n] for n in output_names)


    MAX_TRT_FRAMES = 3000  # split text if estimated frames exceed this (safe under 4000)

    @staticmethod
    def _split_sentences(text: str) -> list:
        """Split text at sentence boundaries for chunked synthesis."""
        import re
        parts = re.split(r'(?<=[。！？\n])(?![。！？\n])|(?<=[.!?\n])(?![.!?\n])', text)
        chunks, buf = [], ""
        for part in parts:
            if len(buf) + len(part) < 10:
                buf += part
            else:
                if buf.strip():
                    chunks.append(buf.strip())
                buf = part
        if buf.strip():
            chunks.append(buf.strip())
        return chunks if len(chunks) > 1 else [text]

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def _synthesize_chunk(self, text: str, noise_scale: float = 0.667):
        """Synthesize a single text chunk, returning float32 numpy array."""
        t0 = time.perf_counter()
        # ── 1. Text → phoneme IDs ──
        norm_text, phones, tones, langs, word2ph = self._clean_text(text)
        phone_ids, tone_ids, lang_ids = self._seq_mix(phones, tones, langs)
        phone_ids = [0] + [p for pid in phone_ids for p in (pid, 0)]
        tone_ids = [0] + [t for tid in tone_ids for t in (tid, 0)]
        lang_ids = [0] + [l for lid in lang_ids for l in (lid, 0)]
        T = len(phone_ids)
        t1 = time.perf_counter()

        ph = np.array([phone_ids], dtype=np.int32)
        to = np.array([tone_ids], dtype=np.int32)
        la = np.array([lang_ids], dtype=np.int32)
        xl = np.array([T], dtype=np.int32)

        # ── 2. Encoder (ORT) ──
        m_p, logs_p, logw, x_mask = self._encoder.run(None,
            {"ph": ph, "to": to, "la": la, "xl": xl})
        t2 = time.perf_counter()

        # ── 3. Duration → expanded frames ──
        w = np.exp(logw[0, 0, :T]) * x_mask[0, 0, :T]
        w_ceil = np.ceil(w).astype(np.int32)
        Ty = int(w_ceil.sum())
        y_mask = np.ones((1, 1, Ty), dtype=np.float32)

        dur = np.maximum(w_ceil, 1).astype(np.int32)
        cumsum = np.concatenate([[0], np.cumsum(dur)[:-1]])
        m_p_exp = np.zeros((1, 256, Ty), dtype=np.float32)
        logs_p_exp = np.zeros((1, 256, Ty), dtype=np.float32)
        for i, (d, pos) in enumerate(zip(dur, cumsum)):
            if d > 0 and pos < Ty:
                end = min(pos + d, Ty)
                m_p_exp[0, :, pos:end] = m_p[0, :, i:i + 1]
                logs_p_exp[0, :, pos:end] = logs_p[0, :, i:i + 1]

        noise_scale = 0.667
        z_p = (m_p_exp +
               np.random.randn(1, 256, Ty).astype(np.float32) *
               np.exp(logs_p_exp) * noise_scale)
        t3 = time.perf_counter()

        # ── 4. Flow (TRT GPU, fallback to ORT if too long) ──
        flow_max = self._flow_eng.get_tensor_profile_shape("z_p", 0)[2][2]
        if z_p.shape[2] > flow_max:
            log.warning("[tts] z_p len %d exceeds TRT flow max %d, using ORT fallback", z_p.shape[2], flow_max)
            import onnxruntime as _ort
            if not hasattr(self, '_flow_ort'):
                self._flow_ort = _ort.InferenceSession(
                    os.path.join(self._trt_dir, "flow.onnx"),
                    providers=["CPUExecutionProvider"])
            z = self._flow_ort.run(None, {"z_p": z_p.astype(np.float32), "y_mask": y_mask.astype(np.float32)})[0]
        else:
            z, = self._trt_run(self._flow_eng,
                                {"z_p": z_p, "y_mask": y_mask}, ["z"])
        t4 = time.perf_counter()

        # ── 5. Decoder (TRT GPU, fallback to ORT if too long) ──
        dec_max = self._dec_eng.get_tensor_profile_shape("z", 0)[2][2]
        if z.shape[2] > dec_max:
            log.warning("[tts] z len %d exceeds TRT decoder max %d, using ORT fallback", z.shape[2], dec_max)
            import onnxruntime as _ort
            if not hasattr(self, '_dec_ort'):
                self._dec_ort = _ort.InferenceSession(
                    os.path.join(self._trt_dir, "decoder_spec.onnx"),
                    providers=["CPUExecutionProvider"])
            spec, phase = self._dec_ort.run(None, {"z": z.astype(np.float32)})
        else:
            spec, phase = self._trt_run(self._dec_eng, {"z": z},
                                         ["spec", "phase"])
        t5 = time.perf_counter()

        # ── 6. iSTFT (NumPy, vectorized) ──
        # Read n_fft from config (default 64 for G_opt_v8, 16 for original)
        n_fft = getattr(hps.model, 'gen_istft_n_fft', 64)
        hop = getattr(hps.model, 'gen_istft_hop_size', 16)
        tf = np.fft.irfft(spec * np.exp(1j * phase), n=n_fft, axis=1)
        window = np.hanning(n_fft).astype(np.float32).reshape(1, n_fft, 1)
        windowed = tf * window  # [1, n_fft, T_frames]
        _, _, T_frames = tf.shape
        out_len = (T_frames - 1) * hop + n_fft
        # Vectorized overlap-add via np.add.at
        audio = np.zeros((1, out_len), dtype=np.float32)
        idx = np.arange(T_frames) * hop  # [0, 4, 8, ...]
        # for each position i in n_fft, add shifted windowed
        for k in range(n_fft):
            np.add.at(audio[0], idx + k, windowed[0, k, :])
        audio = audio[:, n_fft // 2:out_len - n_fft // 2]

        # Apply speed (length_scale) — vectorized linear interpolation
        if self._speed and self._speed != 1.0:
            target_len = int(audio.shape[1] / self._speed)
            xp = np.linspace(0, 1, audio.shape[1])
            x = np.linspace(0, 1, target_len)
            audio = np.interp(x, xp, audio[0]).reshape(1, -1).astype(np.float32)

        # ── 7. Convert to PCM bytes (vectorized) ──
        audio_f32 = audio[0]
        audio_i16 = np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16)
        pcm = audio_i16.tobytes()
        t6 = time.perf_counter()

        dt_total = (t6 - t0) * 1000
        audio_s = len(audio_f32) / 16000
        log.info("[tts] chunk %d chars %d phones → Ty=%d T_frames=%d | "
                 "text=%dms enc(ORT)=%dms mas=%dms flow(TRT)=%dms dec(TRT)=%dms istft+pcm=%dms | "
                 "total=%dms audio=%.1fs",
                 len(text), T, Ty, T_frames,
                 int((t1-t0)*1000), int((t2-t1)*1000), int((t3-t2)*1000),
                 int((t4-t3)*1000), int((t5-t4)*1000), int((t6-t5)*1000),
                 int(dt_total), audio_s)

        return audio_f32

    def synthesize_stream(self, text: str):
        # Split long text into chunks to stay within TRT engine profile limits
        chunks = self._split_sentences(text)
        for chunk_text in chunks:
            audio_f32 = self._synthesize_chunk(chunk_text)
            pcm = np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16).tobytes()
            for i in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[i:i + CHUNK_BYTES]


def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    model_dir = cfg.get("model_dir", "/models/vits2-mix")
    trt_dir = cfg.get("trt_dir", os.path.join(model_dir, "trt"))
    speaker_id = int(cfg.get("speaker_id", 0))
    speed = float(cfg.get("speed", 1.0))
    return TRTTSAdapter(model_dir, trt_dir, speaker_id, speed)


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class _TTSNode(Node):
    def __init__(self, input_topic, adapter, node_suffix=''):
        node_name = f"tts_{node_suffix}" if node_suffix else "tts"
        super().__init__(node_name)
        self._input_topic = input_topic or ''
        self._output_topic = f"{input_topic}/tts" if input_topic else '/perception/tts'
        self._adapter = adapter
        self.state = "idle"
        self._text_queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        from audio_msgs.msg import AudioChunk
        self._pub = self.create_publisher(AudioChunk, self._output_topic, _LOW_LAT_QOS)
        self._sub = self.create_subscription(String, self._input_topic, self._text_cb, _LOW_LAT_QOS) if input_topic else None

    def start(self):
        while not self._text_queue.empty():
            try: self._text_queue.get_nowait()
            except: break
        if self.state == "running": return self._status_dict()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self._status_dict()

    def stop(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def enqueue(self, text: str):
        if self.state != "running": raise RuntimeError("TTS not running")
        self._text_queue.put(text)

    def _text_cb(self, msg):
        if self.state != "running": return
        try: text = json.loads(msg.data).get("text", "")
        except: text = msg.data.strip()
        if text: self._text_queue.put(text)

    def _worker(self):
        from audio_msgs.msg import AudioChunk
        FRAME = CHUNK_BYTES / (SAMPLE_RATE * 2)
        while not self._stop_event.is_set():
            try: text = self._text_queue.get(timeout=1)
            except queue.Empty: continue
            try:
                t0 = time.monotonic()
                total, buf, played, frames = 0, b'', None, 0
                prebuf = []
                for chunk in self._adapter.synthesize_stream(text):
                    if self._stop_event.is_set(): break
                    buf += chunk; total += len(chunk)
                    while len(buf) >= CHUNK_BYTES:
                        frame, buf = buf[:CHUNK_BYTES], buf[CHUNK_BYTES:]
                        if played is None:
                            prebuf.append(frame)
                            if len(prebuf) >= 3:
                                played = time.monotonic()
                                for pf in prebuf:
                                    m = AudioChunk(); m.format = "audio/pcm-16k"
                                    m.data = list(pf); self._pub.publish(m); frames += 1
                                prebuf = []
                            continue
                        m = AudioChunk(); m.format = "audio/pcm-16k"
                        m.data = list(frame); self._pub.publish(m); frames += 1
                if prebuf:
                    for pf in prebuf:
                        m = AudioChunk(); m.format = "audio/pcm-16k"
                        m.data = list(pf); self._pub.publish(m)
                if buf:
                    m = AudioChunk(); m.format = "audio/pcm-16k"
                    m.data = list(buf); self._pub.publish(m)
            except Exception as e:
                log.error(f"[tts] error: {e}", exc_info=True)

    def _status_dict(self):
        return {"state": self.state,
                "topic_in":  [{"topic": self._input_topic,  "format": "data/json"}],
                "topic_out": [{"topic": self._output_topic, "format": "audio/pcm-16k"}]}


# ── Plugin ────────────────────────────────────────────────────────────────────

class TTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg, executor):
        self._cfg = plugin_cfg
        self._loading = False
        self._load_error = None
        self._adapter = None
        try: self._adapter = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] model load failed: {e}", exc_info=True)
            self._adapter = None; self._load_error = str(e)
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        self._nodes = {}
        self._executor = executor

    def get_tools(self): return TOOLS

    def dispatch(self, name, args):
        action = args.get("action") if name == "tts" else name
        iid = args.get("instance_id", "")

        if action == "info":
            return {"name": "TTS", "manufacture": "Embodied", "model": "vits2-int8",
                    "state": "running" if self._nodes else "idle",
                    "topic_in": [{"topic": n._input_topic, "format": "data/json"} for n in self._nodes.values()],
                    "topic_out": [{"topic": n._output_topic, "format": "audio/pcm-16k"} for n in self._nodes.values()]}

        if action == "start":
            input_topic = args.get("input_topic") or ''
            key = iid or input_topic or '_default'
            if key not in self._nodes:
                node = _TTSNode(input_topic or None, self._adapter,
                                node_suffix=key.replace('/', '_').replace('-', '_'))
                self._executor.add_node(node)
                self._nodes[key] = node
            return self._nodes[key].start()

        if action == "stop":
            if iid and iid in self._nodes:
                self._nodes[iid].stop()
                self._executor.remove_node(self._nodes[iid])
                del self._nodes[iid]
            elif not iid:
                for k in list(self._nodes.keys()):
                    self._nodes[k].stop()
                    self._executor.remove_node(self._nodes[k])
                    del self._nodes[k]
            return {"state": "idle"}

        if action == "speak":
            text = args.get("text", "")
            if not text: raise ValueError("text required")
            key = iid or '_default'
            if key not in self._nodes:
                node = _TTSNode(args.get("input_topic") or None, self._adapter,
                                node_suffix=key.replace('/', '_').replace('-', '_'))
                self._executor.add_node(node)
                self._nodes[key] = node
            else: node = self._nodes[key]
            if node.state != "running": node.start()
            node.enqueue(text)
            return {"status": "queued", "text": text}

        if action == "config":
            if 'speaker_id' in args: self._cfg['speaker_id'] = int(args['speaker_id'])
            if 'speed' in args: self._cfg['speed'] = float(args['speed'])
            # ── stop all nodes & release old model before rebuild ────
            for k in list(self._nodes.keys()):
                self._nodes[k].stop()
                self._executor.remove_node(self._nodes[k])
                del self._nodes[k]
            if self._adapter is not None:
                if hasattr(self._adapter, '_net'):
                    del self._adapter._net
                del self._adapter
                self._adapter = None
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            self._adapter = _build_tts_adapter(self._cfg)
            return {"status": "configured"}

        return None

    def synthesize_raw(self, text):
        if not self._adapter: raise RuntimeError("TTS not loaded")
        return self._adapter.synthesize(text)
