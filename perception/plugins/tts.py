#!/usr/bin/env python3
"""
plugins/tts.py — TTSPlugin: VITS2-Mix INT8 PyTorch TTS.

Chinese-English mixed TTS using VITS2-Mix with INT8 quantized weights.
Model: G_B_final_int8.pth (35.4MB, trained on 柒小白 + mixed CN-EN data).
"""

from __future__ import annotations

import json, logging, os, queue, struct, sys, threading, time, types
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

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


class Vits2Int8Adapter(TTSAdapter):
    """VITS2-Mix INT8 PyTorch TTS adapter for G_B_final_int8.pth."""

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0):
        from utils.model_downloader import ensure_model
        ensure_model("vits2", model_dir)

        _model_path = os.path.join(model_dir, "G_B_final_int8.pth")
        _config_path = os.path.join(model_dir, "config.json")

        _vits2_path = os.path.join(model_dir, "vits2_src")
        if os.path.isdir(_vits2_path):
            sys.path.insert(0, _vits2_path)
        from vits2 import models, commons
        from vits2.text import symbols
        from vits2 import utils as vits2_utils

        hps = vits2_utils.get_hparams_from_file(_config_path)

        net = models.SynthesizerTrn(
            len(symbols), hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=1, mas_noise_scale_initial=0.01,
            noise_scale_delta=2e-6, **hps.model)

        ckpt = torch.load(_model_path, map_location="cpu")
        qmodel, qscales = ckpt["model"], ckpt["scales"]

        dq = {}
        for name, tensor in qmodel.items():
            if tensor.dtype == torch.float16:
                dq[name] = tensor.float()
            elif tensor.dtype == torch.int8:
                if name in qscales:
                    s_val = qscales[name]
                    s = torch.tensor(list(s_val) if isinstance(s_val, (list, tuple)) else float(s_val)).float()
                    if s.ndim > 0 and tensor.ndim >= 2:
                        s = s.view(-1, *([1] * (tensor.ndim - 1)))
                    dq[name] = tensor.float() * s
                else:
                    dq[name] = tensor.float()
            else:
                dq[name] = tensor

        net.load_state_dict(dq, strict=False)
        self._net = net.cuda().eval()
        self._hps = hps
        self._commons = commons
        self._spk = torch.tensor([speaker_id], dtype=torch.long, device="cuda")
        self._speed = speed

        _frontend_path = os.path.join(model_dir, "frontend")
        if os.path.isdir(_frontend_path):
            sys.path.insert(0, _frontend_path)
        from frontend.cleaner import clean_text_mix
        from frontend import cleaned_text_to_sequence_mix
        self._clean_text = clean_text_mix
        self._seq_mix = cleaned_text_to_sequence_mix

        log.info(f"[tts] VITS2 INT8 loaded: {_model_path}, "
                 f"params={sum(p.numel() for p in net.parameters())/1e6:.1f}M")

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        norm_text, phones, tones, langs, word2ph = self._clean_text(text)
        phone_ids, tone_ids, lang_ids = self._seq_mix(phones, tones, langs)
        phone_ids = self._commons.intersperse(phone_ids, 0)
        tone_ids = self._commons.intersperse(tone_ids, 0)
        lang_ids = self._commons.intersperse(lang_ids, 0)

        x = torch.tensor([phone_ids], dtype=torch.long, device="cuda")
        t = torch.tensor([tone_ids], dtype=torch.long, device="cuda")
        l = torch.tensor([lang_ids], dtype=torch.long, device="cuda")
        xl = torch.tensor([len(phone_ids)], dtype=torch.long, device="cuda")

        with torch.no_grad():
            audio = self._net.infer(x, xl, self._spk, t, l,
                                     noise_scale=0.667, noise_scale_w=0.8,
                                     length_scale=1.0 / self._speed if self._speed else 1.0)[0][0, 0]

        audio = audio.float().cpu()
        pcm = struct.pack(f'<{len(audio)}h',
                          *[int(max(-32768, min(32767, s * 32767))) for s in audio.tolist()])
        for i in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[i:i + CHUNK_BYTES]


def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    model_dir = cfg.get("model_dir", "/models/vits2-mix")
    speaker_id = int(cfg.get("speaker_id", 0))
    speed = float(cfg.get("speed", 1.0))
    return Vits2Int8Adapter(model_dir, speaker_id, speed)


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
                        target = played + frames * FRAME
                        now = time.monotonic()
                        if now < target: time.sleep(target - now)
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
        try: self._adapter = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] model load failed: {e}", exc_info=True)
            self._adapter = None; self._load_error = str(e)
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
            self._adapter = _build_tts_adapter(self._cfg)
            for k in list(self._nodes.keys()):
                self._nodes[k].stop()
                self._executor.remove_node(self._nodes[k])
                del self._nodes[k]
            return {"status": "configured"}

        return None

    def synthesize_raw(self, text):
        if not self._adapter: raise RuntimeError("TTS not loaded")
        return self._adapter.synthesize(text)
