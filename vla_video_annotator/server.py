#!/usr/bin/env python3
"""Minimal web server for multi-view SAM video annotation.

This file intentionally keeps the server small:
- standard-library HTTP server
- in-process SAM predictor cache keyed by GPU
- background task queue with per-view GPU parallelism
- explicit result IO compatible with the existing frontend
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import json
import mimetypes
import os
import posixpath
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_VIDEO_ROOT = Path(os.environ.get("VLA_ANNOTATOR_VIDEO_ROOT", "data/videos"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("VLA_ANNOTATOR_OUTPUT_ROOT", "outputs/annotations"))
DEFAULT_SAM21_CHECKPOINT = Path(os.environ.get("SAM21_CHECKPOINT", "checkpoints/sam2.1_hiera_base_plus.pt"))
DEFAULT_SAM31_CHECKPOINT = Path(os.environ.get("SAM31_CHECKPOINT", "checkpoints/sam3.1_multiplex.pt"))
DEFAULT_SAM21_CONFIG = os.environ.get("SAM21_CONFIG", "configs/sam2.1/sam2.1_hiera_b+.yaml")
VIEW_DIRS = {
    "head": "observation.images.ros_head",
    "left": "observation.images.ros_left",
    "right": "observation.images.ros_right",
}
LABELS = ["manip_obj", "target"]
COLORS = {
    "manip_obj": (255, 74, 88),
    "target": (50, 184, 112),
}
RUNNING_VIEW_STATES = {"queued", "preparing_frames", "loading_model", "prompting", "propagating", "saving"}
SERVER_TIMELINE = [
    {
        "time": "current",
        "title": "Multi-view queue",
        "detail": "Episode tasks can run selected views across one or more GPU IDs while reusing loaded predictors.",
    },
    {
        "time": "current",
        "title": "SAM2.1 by default",
        "detail": "Point and box prompts use the SAM2.1 video predictor by default. SAM3.1 is available as an optional backend.",
    },
    {
        "time": "current",
        "title": "Preview and propagate",
        "detail": "Preview validates the selected start frame; propagation writes masks from the prompt frame forward.",
    },
]


def resolve_video_root(dataset_root: Path | None, video_root: Path | None) -> Path:
    root = (dataset_root or video_root or DEFAULT_VIDEO_ROOT).expanduser().resolve()
    if root.name == "videos":
        return root
    if (root / "videos").exists():
        return root / "videos"
    return root


class AppState:
    def __init__(
        self,
        video_root: Path,
        output_root: Path,
        cache_root: Path,
        checkpoint: Path | None,
        bpe_path: Path | None,
        model_backend: str,
        sam21_config: str,
        gpu_id: int | None,
        inference_dtype: str,
        async_loading_frames: bool,
        frame_extract_threads: int,
    ) -> None:
        self.video_root = video_root.resolve()
        self.output_root = output_root.resolve()
        self.cache_root = cache_root.resolve()
        self.checkpoint = checkpoint.resolve() if checkpoint else None
        self.bpe_path = bpe_path.resolve() if bpe_path else None
        self.model_backend = model_backend
        self.sam21_config = sam21_config
        self.gpu_id = gpu_id
        self.inference_dtype = inference_dtype
        self.async_loading_frames = async_loading_frames
        self.frame_extract_threads = max(1, int(frame_extract_threads))
        self.predictor = None
        self.predictors: dict[str, Any] = {}
        self.model_status = "not_loaded"
        self.model_error: str | None = None
        self.predictor_lock = threading.Lock()
        self.predictor_locks: dict[str, threading.Lock] = {}
        self.inference_lock = threading.RLock()
        self.inference_locks: dict[str, threading.RLock] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.task_queue: list[dict[str, Any]] = []
        self.task_queue_lock = threading.Lock()
        self.task_runner_active = False
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def safe_video_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        try:
            path.relative_to(self.video_root)
        except ValueError as exc:
            raise ValueError(f"path is outside video root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    def episode_paths(self, episode: str, chunk: str = "chunk-000") -> dict[str, Path]:
        paths = {}
        for view, view_dir in VIEW_DIRS.items():
            path = self.video_root / chunk / view_dir / f"{episode}.mp4"
            if path.exists():
                paths[view] = path.resolve()
        return paths


STATE: AppState


def json_id(prefix: str) -> str:
    return hashlib.sha1(f"{prefix}:{time.time_ns()}".encode()).hexdigest()[:12]


def result_dir(episode: str, view: str) -> Path:
    return STATE.output_root / episode / view


def default_checkpoint_for_backend(model_backend: str) -> Path:
    if model_backend == "sam31":
        return DEFAULT_SAM31_CHECKPOINT
    return DEFAULT_SAM21_CHECKPOINT


def annotation_done(episode: str, view: str) -> bool:
    out_dir = result_dir(episode, view)
    return all((out_dir / name).exists() for name in ("metadata.json", "mask_int.npy", "object_masks.npz"))


def run_json(cmd: list[str]) -> dict[str, Any]:
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def media_info(path: Path) -> dict[str, Any]:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = data["streams"][0]
    num, den = [float(x) for x in stream.get("r_frame_rate", "0/1").split("/")]
    fps = num / den if den else 0.0
    duration = float(stream.get("duration") or 0.0)
    frame_count = int(stream.get("nb_frames") or round(duration * fps) or 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
        "frame_count": max(1, frame_count),
    }


def frame_cache_dir(path: Path) -> Path:
    stat = path.stat()
    key = hashlib.sha1(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    return STATE.cache_root / "frames" / key


def ensure_frame_dir(path: Path, expected_frames: int | None = None) -> Path:
    out_dir = frame_cache_dir(path)
    marker = out_dir / ".complete.json"
    if marker.exists():
        return out_dir
    tmp_marker = out_dir / ".extracting"
    out_dir.mkdir(parents=True, exist_ok=True)
    if tmp_marker.exists() and time.time() - tmp_marker.stat().st_mtime < 600:
        for _ in range(600):
            if marker.exists():
                return out_dir
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for frame extraction: {path}")
    tmp_marker.write_text(str(time.time()), encoding="utf-8")
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            str(STATE.frame_extract_threads),
            "-y",
            "-i",
            str(path),
            "-q:v",
            "2",
            str(out_dir / "%06d.jpg"),
        ],
        check=True,
    )
    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {path}")
    tmp_marker.unlink(missing_ok=True)
    marker.write_text(
        json.dumps({"source": str(path), "frames": len(frames), "expected_frames": expected_frames}),
        encoding="utf-8",
    )
    return out_dir


def ensure_frame(path: Path, frame_idx: int) -> Path:
    out_dir = ensure_frame_dir(path)
    frame = out_dir / f"{frame_idx + 1:06d}.jpg"
    if not frame.exists():
        raise FileNotFoundError(f"missing extracted frame {frame_idx}: {frame}")
    return frame


def list_gpus() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": int(parts[0]), "name": parts[1], "memory_total_mb": int(parts[2])})
    return gpus


def gpu_key(gpu_id: int | None = None) -> str:
    return "default" if gpu_id is None else str(int(gpu_id))


def resolve_gpu_id(gpu_id: int | None = None) -> int | None:
    return STATE.gpu_id if gpu_id is None else gpu_id


def predictor_cache_key(gpu_id: int | None = None) -> str:
    return f"{STATE.model_backend}:{gpu_key(resolve_gpu_id(gpu_id))}"


def get_predictor_lock(key: str) -> threading.Lock:
    with STATE.predictor_lock:
        lock = STATE.predictor_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            STATE.predictor_locks[key] = lock
        return lock


def get_inference_lock(gpu_id: int | None = None) -> threading.RLock:
    key = predictor_cache_key(gpu_id)
    with STATE.predictor_lock:
        lock = STATE.inference_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            STATE.inference_locks[key] = lock
        return lock


def get_predictor(gpu_id: int | None = None):
    gpu_id = resolve_gpu_id(gpu_id)
    key = predictor_cache_key(gpu_id)
    lock = get_predictor_lock(key)
    with lock:
        if key in STATE.predictors:
            return STATE.predictors[key]
        STATE.model_status = "loading"
        STATE.model_error = None
        try:
            if STATE.checkpoint is not None and not STATE.checkpoint.exists():
                raise FileNotFoundError(f"checkpoint not found: {STATE.checkpoint}")
            if STATE.bpe_path is not None and not STATE.bpe_path.exists():
                raise FileNotFoundError(f"BPE file not found: {STATE.bpe_path}")

            import torch

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if gpu_id is not None:
                    torch.cuda.set_device(gpu_id)
            if STATE.model_backend == "sam21":
                predictor = Sam21VideoPredictorWrapper(
                    checkpoint_path=STATE.checkpoint,
                    config_file=STATE.sam21_config,
                    gpu_id=gpu_id,
                )
            elif STATE.model_backend == "sam31":
                from sam3.model_builder import build_sam3_multiplex_video_predictor

                kwargs: dict[str, Any] = {
                    "use_fa3": False,
                    "compile": False,
                    "warm_up": False,
                    "async_loading_frames": STATE.async_loading_frames,
                }
                if STATE.checkpoint is not None:
                    kwargs["checkpoint_path"] = str(STATE.checkpoint)
                if STATE.bpe_path is not None:
                    kwargs["bpe_path"] = str(STATE.bpe_path)
                predictor = build_sam3_multiplex_video_predictor(**kwargs)
            else:
                raise ValueError(f"unsupported model backend: {STATE.model_backend}")
            STATE.predictors[key] = predictor
            if STATE.predictor is None:
                STATE.predictor = predictor
            STATE.model_status = "loaded"
            return predictor
        except Exception as exc:
            STATE.model_status = "error"
            STATE.model_error = traceback.format_exc()
            raise RuntimeError(f"failed to load {STATE.model_backend}: {exc}") from exc


def inference_context(gpu_id: int | None = None):
    if STATE.inference_dtype != "bfloat16":
        if gpu_id is None:
            return nullcontext()
        import torch

        return torch.cuda.device(gpu_id) if torch.cuda.is_available() else nullcontext()
    import torch

    if torch.cuda.is_available():
        class CudaInferenceContext:
            def __enter__(self):
                self.device_ctx = torch.cuda.device(gpu_id) if gpu_id is not None else nullcontext()
                self.autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                self.device_ctx.__enter__()
                self.autocast_ctx.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                self.autocast_ctx.__exit__(exc_type, exc, tb)
                self.device_ctx.__exit__(exc_type, exc, tb)

        return CudaInferenceContext()
    return nullcontext()


class Sam21VideoPredictorWrapper:
    def __init__(self, checkpoint_path: Path | None, config_file: str, gpu_id: int | None) -> None:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM2.1 requires CUDA in this workflow")
        self.device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        self.predictor = build_sam2_video_predictor(
            config_file,
            str(checkpoint_path) if checkpoint_path else None,
            device=self.device,
            vos_optimized=False,
        )
        self.sessions: dict[str, dict[str, Any]] = {}

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request["type"]
        if request_type == "start_session":
            session_id = request.get("session_id") or json_id("sam21")
            self.sessions[session_id] = {
                "state": self.predictor.init_state(video_path=request["resource_path"]),
                "resource_path": request["resource_path"],
            }
            return {"session_id": session_id}
        if request_type == "add_prompt":
            session = self.sessions[request["session_id"]]
            frame_idx, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=session["state"],
                frame_idx=int(request["frame_index"]),
                obj_id=int(request.get("obj_id") or 1),
                points=request.get("points"),
                labels=request.get("point_labels"),
                box=request.get("box_xyxy"),
                clear_old_points=True,
                normalize_coords=True,
            )
            return {"frame_index": frame_idx, "outputs": self.format_outputs(obj_ids, mask_logits)}
        if request_type == "reset_session":
            session = self.sessions[request["session_id"]]
            self.predictor.reset_state(session["state"])
            return {"is_success": True}
        if request_type == "close_session":
            self.sessions.pop(request["session_id"], None)
            return {"is_success": True}
        raise RuntimeError(f"unsupported SAM2.1 request type: {request_type}")

    def handle_stream_request(self, request: dict[str, Any]):
        session = self.sessions[request["session_id"]]
        start_frame_idx = request.get("start_frame_index")
        max_frame_num_to_track = request.get("max_frame_num_to_track")
        reverse = request.get("propagation_direction") == "backward"
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
            session["state"],
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        ):
            yield {"frame_index": frame_idx, "outputs": self.format_outputs(obj_ids, mask_logits)}

    @staticmethod
    def format_outputs(obj_ids: Any, mask_logits: Any) -> dict[str, Any]:
        masks = to_numpy(mask_logits)
        if masks.ndim == 4:
            masks = masks[:, 0]
        return {
            "out_obj_ids": np.asarray(obj_ids, dtype=np.int64),
            "out_binary_masks": masks > 0,
        }


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_labels(items: list[dict[str, Any]], prompt_mode: str) -> list[dict[str, Any]]:
    def normalize_prompt_segment(raw_segment: dict[str, Any], fallback_frame: int) -> dict[str, Any] | None:
        frame_index = int(raw_segment.get("frame_index", fallback_frame))
        points = []
        for point in raw_segment.get("points") or []:
            points.append(
                {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "frame": int(point.get("frame", frame_index)),
                    "label": int(point.get("label", 1)),
                }
            )
        box = raw_segment.get("box_xywh") or raw_segment.get("box")
        box_xywh = [float(x) for x in box] if box else None
        if not points and box_xywh is None:
            return None
        return {"frame_index": frame_index, "points": points, "box_xywh": box_xywh}

    labels = []
    for i, raw in enumerate(items, start=1):
        if not raw.get("enabled", True):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        item = dict(raw)
        item["name"] = name
        item["obj_id"] = int(item.get("obj_id") or i)
        item["frame_index"] = int(item.get("frame_index", 0))
        item["text"] = str(item.get("text") or name).strip()
        if prompt_mode == "text":
            item["points"] = []
            box = item.get("box_xywh") or item.get("box")
            item["box_xywh"] = [float(x) for x in box] if box else None
            item["segments"] = []
            if item["text"]:
                labels.append(item)
            continue

        segments = []
        for raw_segment in item.get("segments") or []:
            segment = normalize_prompt_segment(raw_segment, item["frame_index"])
            if segment is not None:
                segments.append(segment)
        fallback_segment = normalize_prompt_segment(item, item["frame_index"])
        if not segments and fallback_segment is not None:
            segments.append(fallback_segment)
        segments.sort(key=lambda x: int(x["frame_index"]))
        item["segments"] = segments
        if segments:
            item["frame_index"] = int(segments[0]["frame_index"])
            item["points"] = segments[0]["points"]
            item["box_xywh"] = segments[0]["box_xywh"]
            labels.append(item)
    if not labels:
        if prompt_mode == "text":
            raise ValueError("at least one enabled label with text is required")
        raise ValueError("at least one enabled label with a point or box is required")
    return labels


def normalize_empty_labels(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = []
    for i, raw in enumerate(items, start=1):
        if not raw.get("enabled", True):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        labels.append(
            {
                **raw,
                "name": name,
                "text": str(raw.get("text") or name).strip(),
                "obj_id": int(raw.get("obj_id") or i),
                "frame_index": int(raw.get("frame_index", 0)),
                "points": [],
                "box_xywh": raw.get("box_xywh"),
            }
        )
    return labels or [{"name": name, "text": name, "obj_id": i + 1, "frame_index": 0, "points": [], "box_xywh": None} for i, name in enumerate(LABELS)]


def merge_prompt_outputs(
    target: dict[int, dict[str, Any]],
    frame_idx: int,
    outputs: Any,
    label_idx: int,
    obj_to_label: dict[int, int],
) -> None:
    if outputs is None:
        return
    out_obj_ids = to_numpy(outputs.get("out_obj_ids", []))
    for oid in out_obj_ids:
        obj_to_label[int(oid)] = label_idx
    merge_known_outputs(target, frame_idx, outputs, obj_to_label)


def merge_known_outputs(
    target: dict[int, dict[str, Any]],
    frame_idx: int,
    outputs: Any,
    obj_to_label: dict[int, int],
) -> None:
    if outputs is None:
        return
    obj_ids = [int(x) for x in to_numpy(outputs.get("out_obj_ids", [])).tolist()]
    masks = outputs.get("out_binary_masks")
    if masks is None or not obj_ids:
        return
    masks_np = to_numpy(masks)
    if masks_np.size == 0:
        return
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    frame = target.setdefault(frame_idx, {"out_obj_ids": [], "out_binary_masks": []})
    for i, obj_id in enumerate(obj_ids):
        label_idx = obj_to_label.get(obj_id)
        if label_idx is None or i >= len(masks_np):
            continue
        mask = masks_np[i]
        if mask.ndim == 3:
            mask = mask[0]
        frame["out_obj_ids"].append(label_idx)
        frame["out_binary_masks"].append(mask > 0 if mask.dtype != np.bool_ else mask)


def build_prompt_request(session_id: str, item: dict[str, Any], prompt_mode: str, width: int, height: int) -> dict[str, Any]:
    frame_idx = int(item.get("frame_index", 0))
    if STATE.model_backend == "sam21":
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "obj_id": int(item["obj_id"]),
        }
        points = item.get("points") or []
        if points:
            request["points"] = [[float(p["x"]), float(p["y"])] for p in points]
            request["point_labels"] = [int(p.get("label", 1)) for p in points]
        if item.get("box_xywh") is not None:
            x, y, w, h = [float(v) for v in item["box_xywh"]]
            request["box_xyxy"] = [x, y, x + w, y + h]
        if "points" not in request and "box_xyxy" not in request:
            raise ValueError("SAM2.1 backend requires a point or box prompt; text-only is not supported")
        return request
    if prompt_mode == "text":
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "text": item["text"],
        }
        if item.get("box_xywh") is not None:
            x, y, w, h = [float(v) for v in item["box_xywh"]]
            request["bounding_boxes"] = [[x / width, y / height, w / width, h / height]]
            request["bounding_box_labels"] = [1]
            request["obj_id"] = int(item["obj_id"])
        return request
    if item.get("box_xywh") is not None:
        x, y, w, h = [float(v) for v in item["box_xywh"]]
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "bounding_boxes": [[x / width, y / height, w / width, h / height]],
            "bounding_box_labels": [1],
            "obj_id": int(item["obj_id"]),
        }
        if item.get("text"):
            request["text"] = item["text"]
        return request
    return {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_idx,
        "points": [[float(p["x"]) / width, float(p["y"]) / height] for p in item["points"]],
        "point_labels": [int(p.get("label", 1)) for p in item["points"]],
        "obj_id": int(item["obj_id"]),
    }


def prompt_segments_for_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    segments = item.get("segments") or []
    if segments:
        return [{**item, **segment} for segment in segments]
    return [item]


def reset_or_restart_session(predictor: Any, session_id: str, frame_resource: Path, sessions: list[str]) -> str:
    try:
        predictor.handle_request({"type": "reset_session", "session_id": session_id})
        return session_id
    except Exception:
        try:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        except Exception:
            pass
        response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_resource)})
        new_session_id = response["session_id"]
        sessions.append(new_session_id)
        return new_session_id


def predict_labels(
    predictor: Any,
    frame_resource: Path,
    labels: list[dict[str, Any]],
    info: dict[str, Any],
    prompt_mode: str,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[int, dict[str, Any]]:
    def update(**kwargs: Any) -> None:
        if progress_cb:
            progress_cb(kwargs)

    width, height = info["width"], info["height"]
    outputs_by_frame: dict[int, dict[str, Any]] = {}
    sessions: list[str] = []
    try:
        response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_resource)})
        session_id = response["session_id"]
        sessions.append(session_id)
        first_segment = True
        for label_idx, item in enumerate(labels, start=1):
            segments = prompt_segments_for_item(item)
            for segment_idx, segment in enumerate(segments):
                if first_segment:
                    first_segment = False
                else:
                    session_id = reset_or_restart_session(predictor, session_id, frame_resource, sessions)
                frame_idx = int(segment.get("frame_index", 0))
                next_frame = int(segments[segment_idx + 1]["frame_index"]) if segment_idx + 1 < len(segments) else int(info["frame_count"])
                end_frame = max(frame_idx + 1, min(int(info["frame_count"]), next_frame))
                update(
                    status="prompting",
                    message=f'{item["name"]}: adding prompt on frame {frame_idx}',
                    progress=((label_idx - 1) / max(1, len(labels))) * 0.95,
                )
                request = build_prompt_request(session_id, segment, prompt_mode, width, height)
                prompt_response = predictor.handle_request(request)
                obj_to_label = {int(item["obj_id"]): label_idx}
                prompt_frame = int(prompt_response["frame_index"])
                if frame_idx <= prompt_frame < end_frame:
                    merge_prompt_outputs(
                        outputs_by_frame,
                        prompt_frame,
                        prompt_response.get("outputs"),
                        label_idx,
                        obj_to_label,
                    )
                update(status="propagating", message=f'{item["name"]}: propagating {frame_idx}-{end_frame - 1}')
                max_frames = max(1, end_frame - frame_idx)
                stream_request = {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": frame_idx,
                    "max_frame_num_to_track": max_frames,
                }
                for n, stream_response in enumerate(
                    predictor.handle_stream_request(stream_request)
                ):
                    out_frame = int(stream_response["frame_index"])
                    if out_frame < frame_idx or out_frame >= end_frame:
                        continue
                    merge_known_outputs(
                        outputs_by_frame,
                        out_frame,
                        stream_response.get("outputs"),
                        obj_to_label,
                    )
                    segment_progress = (segment_idx + min(1.0, n / max_frames)) / max(1, len(segments))
                    done = ((label_idx - 1) + segment_progress) / max(1, len(labels))
                    update(progress=min(0.95, done * 0.95))
    finally:
        for session_id in sessions:
            try:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                pass
    return outputs_by_frame


def preview_labels(
    predictor: Any,
    frame_resource: Path,
    labels: list[dict[str, Any]],
    info: dict[str, Any],
    prompt_mode: str,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[int, dict[str, Any]]:
    def update(**kwargs: Any) -> None:
        if progress_cb:
            progress_cb(kwargs)

    width, height = info["width"], info["height"]
    outputs_by_frame: dict[int, dict[str, Any]] = {}
    for label_idx, item in enumerate(labels, start=1):
        response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_resource)})
        session_id = response["session_id"]
        try:
            frame_idx = int(item.get("frame_index", 0))
            update(
                status="prompting",
                message=f'{item["name"]}: preview prompt on frame {frame_idx}',
                progress=((label_idx - 1) / max(1, len(labels))) * 0.95,
            )
            prompt_response = predictor.handle_request(build_prompt_request(session_id, item, prompt_mode, width, height))
            obj_to_label = {int(item["obj_id"]): label_idx}
            merge_prompt_outputs(
                outputs_by_frame,
                int(prompt_response["frame_index"]),
                prompt_response.get("outputs"),
                label_idx,
                obj_to_label,
            )
        finally:
            try:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                pass
    return outputs_by_frame


def predict_one_view(
    episode: str,
    chunk: str,
    view: str,
    labels: list[dict[str, Any]],
    prompt_mode: str,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    def update(**kwargs: Any) -> None:
        if progress_cb:
            progress_cb(kwargs)

    paths = STATE.episode_paths(episode, chunk)
    if view not in paths:
        raise FileNotFoundError(f"missing {view} video for {episode}")
    video_path = paths[view]
    info = media_info(video_path)
    update(status="preparing_frames", progress=0.0, message=f"{episode}/{view}: preparing frames")
    frame_resource = ensure_frame_dir(video_path, info["frame_count"])
    gpu_text = f" on gpu {gpu_id}" if gpu_id is not None else ""
    update(status="loading_model", progress=0.02, message=f"{episode}/{view}: loading {STATE.model_backend}{gpu_text}")
    lock = get_inference_lock(gpu_id)
    with lock:
        predictor = get_predictor(gpu_id)
        with inference_context(gpu_id):
            outputs_by_frame = predict_labels(predictor, frame_resource, labels, info, prompt_mode, progress_cb)
    update(status="saving", progress=0.98, message=f"{episode}/{view}: saving results")
    return save_prediction(episode, view, video_path, labels, outputs_by_frame, info, prompt_mode)


def preview_one_view(
    episode: str,
    chunk: str,
    view: str,
    labels: list[dict[str, Any]],
    prompt_mode: str,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    def update(**kwargs: Any) -> None:
        if progress_cb:
            progress_cb(kwargs)

    paths = STATE.episode_paths(episode, chunk)
    if view not in paths:
        raise FileNotFoundError(f"missing {view} video for {episode}")
    video_path = paths[view]
    info = media_info(video_path)
    update(status="preparing_frames", progress=0.0, message=f"{episode}/{view}: preparing preview frame")
    frame_resource = ensure_frame_dir(video_path, info["frame_count"])
    gpu_text = f" on gpu {gpu_id}" if gpu_id is not None else ""
    update(status="loading_model", progress=0.02, message=f"{episode}/{view}: loading {STATE.model_backend}{gpu_text}")
    lock = get_inference_lock(gpu_id)
    with lock:
        predictor = get_predictor(gpu_id)
        with inference_context(gpu_id):
            outputs_by_frame = preview_labels(predictor, frame_resource, labels, info, prompt_mode, progress_cb)
    update(status="saving", progress=0.98, message=f"{episode}/{view}: saving preview")
    return save_prediction(episode, view, video_path, labels, outputs_by_frame, info, f"{prompt_mode}_preview")


def save_empty_prediction_for_view(episode: str, chunk: str, view: str, labels: list[dict[str, Any]]) -> dict[str, Any]:
    paths = STATE.episode_paths(episode, chunk)
    if view not in paths:
        raise FileNotFoundError(f"missing {view} video for {episode}")
    info = media_info(paths[view])
    return save_prediction(episode, view, paths[view], labels, {}, info, "empty")


def save_prediction(
    episode: str,
    view: str,
    video_path: Path,
    labels: list[dict[str, Any]],
    outputs_by_frame: dict[int, dict[str, Any]],
    info: dict[str, Any],
    prompt_mode: str,
) -> dict[str, Any]:
    out_dir = result_dir(episode, view)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir / "overlays", ignore_errors=True)
    label_names = [x["name"] for x in labels]
    frame_count, height, width = int(info["frame_count"]), int(info["height"]), int(info["width"])
    mask_int = np.zeros((frame_count, height, width), dtype=np.uint16)
    boxes = np.full((frame_count, len(label_names), 4), np.nan, dtype=np.float32)
    object_masks = {name: np.zeros((frame_count, height, width), dtype=np.bool_) for name in label_names}
    total_masks = 0
    kept_masks = 0

    for frame_idx, outputs in outputs_by_frame.items():
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        obj_ids = [int(x) for x in to_numpy(outputs.get("out_obj_ids", [])).tolist()]
        masks = outputs.get("out_binary_masks")
        if masks is None:
            masks = []
        for obj_id, mask in zip(obj_ids, masks):
            total_masks += 1
            if obj_id < 1 or obj_id > len(label_names):
                continue
            mask_np = to_numpy(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            mask_bool = mask_np.astype(bool)
            if mask_bool.shape != (height, width):
                from PIL import Image

                mask_bool = np.asarray(
                    Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
                ) > 0
            name = label_names[obj_id - 1]
            object_masks[name][frame_idx] |= mask_bool
            merged = object_masks[name][frame_idx]
            mask_int[frame_idx][merged] = obj_id
            if merged.any():
                ys, xs = np.where(merged)
                boxes[frame_idx, obj_id - 1] = [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() - xs.min() + 1),
                    float(ys.max() - ys.min() + 1),
                ]
            kept_masks += 1

    np.save(out_dir / "mask_int.npy", mask_int)
    np.save(out_dir / "boxes_xywh.npy", boxes)
    np.savez_compressed(out_dir / "object_masks.npz", **object_masks)
    prompts = [
        {
            "name": item["name"],
            "obj_id": int(item.get("obj_id", i + 1)),
            "text": str(item.get("text", "")),
            "frame_index": int(item.get("frame_index", 0)),
            "points": [
                {
                    "frame": int(point.get("frame", item.get("frame_index", 0))),
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "label": int(point.get("label", 1)),
                }
                for point in item.get("points", [])
            ],
            "box_xywh": [float(x) for x in item["box_xywh"]] if item.get("box_xywh") is not None else None,
            "segments": [
                {
                    "frame_index": int(segment.get("frame_index", item.get("frame_index", 0))),
                    "points": [
                        {
                            "frame": int(point.get("frame", segment.get("frame_index", item.get("frame_index", 0)))),
                            "x": float(point["x"]),
                            "y": float(point["y"]),
                            "label": int(point.get("label", 1)),
                        }
                        for point in segment.get("points", [])
                    ],
                    "box_xywh": [float(x) for x in segment["box_xywh"]] if segment.get("box_xywh") is not None else None,
                }
                for segment in item.get("segments", [])
            ],
        }
        for i, item in enumerate(labels)
    ]
    (out_dir / "prompts.json").write_text(
        json.dumps({"episode": episode, "view": view, "video_path": str(video_path), "labels": prompts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    nonzero_frames = np.flatnonzero(mask_int.reshape(frame_count, -1).any(axis=1))
    per_label = {}
    for name, masks in object_masks.items():
        frames = np.flatnonzero(masks.reshape(frame_count, -1).any(axis=1))
        per_label[name] = {
            "nonzero_frame_count": int(len(frames)),
            "first_nonzero_frame": int(frames[0]) if len(frames) else None,
            "last_nonzero_frame": int(frames[-1]) if len(frames) else None,
        }
    metadata = {
        "episode": episode,
        "view": view,
        "video_path": str(video_path),
        "prompt_mode": prompt_mode,
        "labels": label_names,
        "prompts": prompts,
        "label_ids": {name: i + 1 for i, name in enumerate(label_names)},
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "prediction_stats": {
            "output_frame_count": int(len(outputs_by_frame)),
            "total_output_masks": int(total_masks),
            "kept_output_masks": int(kept_masks),
            "nonzero_frame_count": int(len(nonzero_frames)),
            "first_nonzero_frame": int(nonzero_frames[0]) if len(nonzero_frames) else None,
            "last_nonzero_frame": int(nonzero_frames[-1]) if len(nonzero_frames) else None,
            "per_label": per_label,
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "mask_int": str(out_dir / "mask_int.npy"),
            "boxes_xywh": str(out_dir / "boxes_xywh.npy"),
            "object_masks": str(out_dir / "object_masks.npz"),
            "prompts": str(out_dir / "prompts.json"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    start_render_overlays(episode, view, [int(x) for x in nonzero_frames[:80]])
    return metadata


def render_overlay(episode: str, view: str, frame_idx: int) -> Path:
    paths = STATE.episode_paths(episode)
    if view not in paths:
        raise FileNotFoundError(f"missing {view} video for {episode}")
    out_dir = result_dir(episode, view)
    mask_path = out_dir / "mask_int.npy"
    meta_path = out_dir / "metadata.json"
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out = overlay_dir / f"{frame_idx:06d}.png"
    if out.exists() and out.stat().st_mtime_ns > mask_path.stat().st_mtime_ns:
        return out
    from PIL import Image, ImageDraw

    frame = Image.open(ensure_frame(paths[view], frame_idx)).convert("RGBA")
    mask = np.load(mask_path, mmap_mode="r")[frame_idx]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for label_idx, name in enumerate(meta["labels"], start=1):
        ys, xs = np.where(mask == label_idx)
        if len(xs) == 0:
            continue
        color = COLORS.get(name, (80, 150, 255))
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask == label_idx] = (*color, 95)
        overlay.alpha_composite(Image.fromarray(rgba, "RGBA"))
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 230), width=3)
        draw.text((x0 + 4, max(0, y0 - 18)), name, fill=(*color, 255))
    frame.alpha_composite(overlay)
    frame.convert("RGB").save(out)
    return out


def start_render_overlays(episode: str, view: str, frame_indices: list[int]) -> None:
    def run() -> None:
        for frame_idx in frame_indices:
            try:
                render_overlay(episode, view, frame_idx)
            except Exception:
                traceback.print_exc()
                return

    threading.Thread(target=run, daemon=True).start()


def encode_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    flat = mask.astype(np.uint8).reshape(-1)
    counts = []
    last = 0
    run = 0
    for raw in flat:
        value = int(raw)
        if value == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = value
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def decode_mask_rle(rle: dict[str, Any]) -> np.ndarray:
    h, w = [int(x) for x in rle["size"]]
    values = []
    value = 0
    for count in rle["counts"]:
        values.extend([value] * int(count))
        value = 1 - value
    arr = np.asarray(values, dtype=np.uint8)
    if arr.size != h * w:
        raise ValueError(f"bad RLE length {arr.size}, expected {h * w}")
    return arr.reshape((h, w)).astype(bool)


def apply_manual_edit(data: dict[str, Any]) -> dict[str, Any]:
    episode, view, label = data["episode"], data["view"], data["label"]
    frame_idx = int(data["frame_index"])
    out_dir = result_dir(episode, view)
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    if label not in meta["labels"]:
        raise ValueError(f"unknown label {label}")
    label_idx = meta["labels"].index(label) + 1
    mask_int = np.load(out_dir / "mask_int.npy")
    boxes = np.load(out_dir / "boxes_xywh.npy")
    object_masks = dict(np.load(out_dir / "object_masks.npz"))
    if "mask_rle" in data:
        mask = decode_mask_rle(data["mask_rle"])
        if mask.shape != mask_int.shape[1:]:
            raise ValueError(f"mask shape {mask.shape} != {mask_int.shape[1:]}")
        mask_int[frame_idx][mask_int[frame_idx] == label_idx] = 0
        mask_int[frame_idx][mask] = label_idx
        object_masks[label][frame_idx] = mask
        if mask.any():
            ys, xs = np.where(mask)
            boxes[frame_idx, label_idx - 1] = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() - xs.min() + 1),
                float(ys.max() - ys.min() + 1),
            ]
        else:
            boxes[frame_idx, label_idx - 1] = np.nan
    if "box_xywh" in data:
        boxes[frame_idx, label_idx - 1] = np.asarray(data["box_xywh"], dtype=np.float32)
    np.save(out_dir / "mask_int.npy", mask_int)
    np.save(out_dir / "boxes_xywh.npy", boxes)
    np.savez_compressed(out_dir / "object_masks.npz", **object_masks)
    with (out_dir / "manual_edits.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    overlay = render_overlay(episode, view, frame_idx)
    return {"ok": True, "overlay_url": f"/api/overlay?episode={episode}&view={view}&frame={frame_idx}&t={overlay.stat().st_mtime_ns}"}


def start_prepare_frames(path: Path) -> dict[str, Any]:
    job_id = json_id("prepare")
    STATE.jobs[job_id] = {"status": "queued", "progress": 0.0, "message": "queued"}

    def run() -> None:
        try:
            info = media_info(path)
            out_dir = ensure_frame_dir(path, info["frame_count"])
            STATE.jobs[job_id].update(status="done", progress=1.0, message=f"prepared {out_dir}", frame_dir=str(out_dir))
        except Exception as exc:
            STATE.jobs[job_id].update(status="error", progress=1.0, message=str(exc), traceback=traceback.format_exc())

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


def start_prediction_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = json_id("predict")
    prompt_mode = payload.get("prompt_mode", "text")
    labels = normalize_labels(payload.get("labels", []), prompt_mode)
    STATE.jobs[job_id] = {"status": "queued", "progress": 0.0, "message": "queued"}

    def progress(update: dict[str, Any]) -> None:
        STATE.jobs[job_id].update(update)

    def run() -> None:
        try:
            metadata = predict_one_view(
                payload["episode"],
                payload.get("chunk", "chunk-000"),
                payload["view"],
                labels,
                prompt_mode,
                progress,
                resolve_gpu_id(None),
            )
            STATE.jobs[job_id].update(status="done", progress=1.0, message="done", metadata=metadata)
        except Exception as exc:
            STATE.jobs[job_id].update(status="error", progress=1.0, message=str(exc), traceback=traceback.format_exc())

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


def start_preview_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = json_id("preview")
    prompt_mode = payload.get("prompt_mode", "text")
    labels = normalize_labels(payload.get("labels", []), prompt_mode)
    STATE.jobs[job_id] = {"status": "queued", "progress": 0.0, "message": "queued"}

    def progress(update: dict[str, Any]) -> None:
        STATE.jobs[job_id].update(update)

    def run() -> None:
        try:
            metadata = preview_one_view(
                payload["episode"],
                payload.get("chunk", "chunk-000"),
                payload["view"],
                labels,
                prompt_mode,
                progress,
                resolve_gpu_id(None),
            )
            STATE.jobs[job_id].update(status="done", progress=1.0, message="preview done", metadata=metadata)
        except Exception as exc:
            STATE.jobs[job_id].update(status="error", progress=1.0, message=str(exc), traceback=traceback.format_exc())

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


def parse_gpu_ids(raw: Any) -> list[int | None]:
    if raw in (None, "", []):
        return [STATE.gpu_id]
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(",") if x.strip()]
    ids: list[int | None] = []
    for item in raw:
        ids.append(None if item in (None, "") else int(item))
    return ids or [STATE.gpu_id]


def queue_snapshot() -> dict[str, Any]:
    with STATE.task_queue_lock:
        items = json.loads(json.dumps(STATE.task_queue, ensure_ascii=False))
    return {
        "status": "running" if any(x["status"] == "running" for x in items) else "idle",
        "total": len(items),
        "done": sum(1 for x in items if x["status"] == "done"),
        "errors": sum(1 for x in items if x["status"] == "error"),
        "running": sum(1 for x in items if x["status"] == "running"),
        "dispatched": 0,
        "queued": sum(1 for x in items if x["status"] == "queued"),
        "items": items,
    }


def update_queue_item(queue_id: str, **updates: Any) -> None:
    with STATE.task_queue_lock:
        for item in STATE.task_queue:
            if item["queue_id"] == queue_id:
                item.update(updates)
                item["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                return


def set_view_status(queue_id: str, view: str, **updates: Any) -> None:
    with STATE.task_queue_lock:
        for item in STATE.task_queue:
            if item["queue_id"] == queue_id:
                item["view_status"][view].update(updates)
                item["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                total = max(1, len(item["views"]))
                item["progress"] = sum(float(v.get("progress", 0.0)) for v in item["view_status"].values()) / total
                messages = [v.get("message", "") for v in item["view_status"].values() if v.get("status") in RUNNING_VIEW_STATES]
                if messages:
                    item["message"] = messages[-1]
                return


def gpu_for_view(index: int, gpu_ids: list[int | None]) -> int | None:
    if not gpu_ids:
        return STATE.gpu_id
    return gpu_ids[index % len(gpu_ids)]


def enqueue_episode(payload: dict[str, Any]) -> dict[str, Any]:
    prompt_mode = payload.get("prompt_mode", "text")
    views = []
    for view_payload in payload.get("views", []):
        view = view_payload.get("view")
        if view not in VIEW_DIRS:
            continue
        empty = bool(view_payload.get("empty", False))
        labels = normalize_empty_labels(view_payload.get("labels", [])) if empty else normalize_labels(view_payload.get("labels", []), prompt_mode)
        views.append({"view": view, "labels": labels, "empty": empty, "frame_index": int(view_payload.get("frame_index", 0))})
    if not payload.get("episode"):
        raise ValueError("episode is required")
    if not views:
        raise ValueError("at least one view payload is required")
    queue_id = json_id("queue")
    item = {
        "queue_id": queue_id,
        "episode": payload["episode"],
        "chunk": payload.get("chunk", "chunk-000"),
        "prompt_mode": prompt_mode,
        "views": views,
        "gpu_ids": parse_gpu_ids(payload.get("gpu_ids")),
        "overwrite": bool(payload.get("overwrite", False)),
        "status": "queued",
        "progress": 0.0,
        "message": "queued",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "view_status": {
            v["view"]: {
                "status": "queued",
                "progress": 0.0,
                "message": "queued",
                "gpu_id": gpu_for_view(i, parse_gpu_ids(payload.get("gpu_ids"))),
            }
            for i, v in enumerate(views)
        },
    }
    with STATE.task_queue_lock:
        STATE.task_queue.append(item)
    ensure_task_runner()
    return {"queue_id": queue_id, "queue": queue_snapshot()}


def ensure_task_runner() -> None:
    with STATE.task_queue_lock:
        if STATE.task_runner_active:
            return
        STATE.task_runner_active = True
    threading.Thread(target=task_runner_loop, daemon=True).start()


def task_runner_loop() -> None:
    try:
        while True:
            with STATE.task_queue_lock:
                item = next((x for x in STATE.task_queue if x["status"] == "queued"), None)
                if item is None:
                    return
                item["status"] = "running"
                item["message"] = "running"
                item["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                item_copy = json.loads(json.dumps(item, ensure_ascii=False))
            run_queue_item(item_copy)
    finally:
        with STATE.task_queue_lock:
            STATE.task_runner_active = False


def run_queue_item(item: dict[str, Any]) -> None:
    queue_id = item["queue_id"]
    gpu_ids = item.get("gpu_ids") or [STATE.gpu_id]

    def run_view(index: int, view_payload: dict[str, Any]) -> dict[str, Any]:
        view = view_payload["view"]
        gpu_id = gpu_for_view(index, gpu_ids)
        set_view_status(queue_id, view, status="queued", progress=0.0, message="queued", gpu_id=gpu_id)
        if annotation_done(item["episode"], view) and not item.get("overwrite", False):
            set_view_status(queue_id, view, status="done", progress=1.0, message="skipped existing result", gpu_id=gpu_id)
            return {"view": view, "skipped": True}

        def progress(update: dict[str, Any], view: str = view, gpu_id: int | None = gpu_id) -> None:
            set_view_status(queue_id, view, gpu_id=gpu_id, **update)

        if view_payload.get("empty"):
            metadata = save_empty_prediction_for_view(item["episode"], item["chunk"], view, view_payload["labels"])
        else:
            metadata = predict_one_view(
                item["episode"],
                item["chunk"],
                view,
                view_payload["labels"],
                item["prompt_mode"],
                progress,
                gpu_id,
            )
        set_view_status(queue_id, view, status="done", progress=1.0, message="done", gpu_id=gpu_id, files=metadata.get("files", {}))
        return {"view": view, "files": metadata.get("files", {})}

    try:
        worker_count = min(len(item["views"]), max(1, len(gpu_ids)))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"queue-{queue_id}") as pool:
            futures = {pool.submit(run_view, i, view_payload): view_payload["view"] for i, view_payload in enumerate(item["views"])}
            errors: list[tuple[str, BaseException]] = []
            for future in as_completed(futures):
                view = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append((view, exc))
                    set_view_status(queue_id, view, status="error", progress=1.0, message=str(exc), traceback=traceback.format_exc())
            if errors:
                raise RuntimeError("; ".join(f"{view}: {exc}" for view, exc in errors))
        update_queue_item(queue_id, status="done", progress=1.0, message="done")
    except Exception as exc:
        update_queue_item(queue_id, status="error", progress=1.0, message=str(exc), traceback=traceback.format_exc())
        for view in item.get("views", []):
            current = queue_snapshot()
            row = next((x for x in current["items"] if x["queue_id"] == queue_id), None)
            if row and row["view_status"].get(view["view"], {}).get("status") != "done":
                set_view_status(queue_id, view["view"], status="error", progress=1.0, message=str(exc), traceback=traceback.format_exc())
                break


def clear_finished_queue_items() -> dict[str, Any]:
    with STATE.task_queue_lock:
        STATE.task_queue = [x for x in STATE.task_queue if x["status"] not in {"done", "error"}]
    return queue_snapshot()


def reset_model_state() -> dict[str, Any]:
    with STATE.inference_lock:
        STATE.predictor = None
        STATE.predictors.clear()
        STATE.model_status = "not_loaded"
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    return {"ok": True, "model_status": STATE.model_status, "predictor_loaded": False}


class Handler(BaseHTTPRequestHandler):
    server_version = "VLAVideoAnnotator/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def query(self) -> dict[str, list[str]]:
        return urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)

    def do_GET(self) -> None:
        try:
            split = urllib.parse.urlsplit(self.path)
            if split.path == "/":
                return self.send_static("index.html")
            if split.path.startswith("/static/"):
                return self.send_static(split.path.removeprefix("/static/"))
            if split.path == "/api/config":
                return self.send_json(self.config_payload())
            if split.path == "/api/episodes":
                return self.handle_episodes()
            if split.path == "/api/job":
                q = self.query()
                return self.send_json(STATE.jobs.get(q["id"][0], {"status": "missing"}))
            if split.path == "/api/task_queue":
                return self.send_json(queue_snapshot())
            if split.path == "/api/server_timeline":
                return self.send_json({"timeline": SERVER_TIMELINE})
            if split.path == "/api/video_info":
                q = self.query()
                return self.send_json(media_info(STATE.safe_video_path(q["path"][0])))
            if split.path == "/api/frame":
                q = self.query()
                return self.send_file(ensure_frame(STATE.safe_video_path(q["path"][0]), int(q.get("frame", ["0"])[0])))
            if split.path == "/api/overlay":
                q = self.query()
                return self.send_file(render_overlay(q["episode"][0], q["view"][0], int(q.get("frame", ["0"])[0])))
            if split.path == "/api/mask_rle":
                q = self.query()
                episode, view, label = q["episode"][0], q["view"][0], q["label"][0]
                frame_idx = int(q.get("frame", ["0"])[0])
                meta = json.loads((result_dir(episode, view) / "metadata.json").read_text(encoding="utf-8"))
                if label not in meta["labels"]:
                    raise ValueError(f"unknown label {label}")
                mask = dict(np.load(result_dir(episode, view) / "object_masks.npz"))[label][frame_idx]
                return self.send_json(encode_mask_rle(mask))
            if split.path == "/media":
                q = self.query()
                return self.send_file(STATE.safe_video_path(q["path"][0]), range_ok=True)
            return self.send_error_json(404, "not found")
        except Exception as exc:
            self.send_error_json(500, f"{exc}\n{traceback.format_exc()}")

    def do_POST(self) -> None:
        try:
            split = urllib.parse.urlsplit(self.path)
            data = self.read_json()
            if split.path == "/api/preview":
                return self.send_json(start_preview_job(data))
            if split.path == "/api/predict":
                return self.send_json(start_prediction_job(data))
            if split.path == "/api/enqueue_episode":
                return self.send_json(enqueue_episode(data))
            if split.path == "/api/clear_finished_queue":
                return self.send_json(clear_finished_queue_items())
            if split.path == "/api/release_model":
                return self.send_json(reset_model_state())
            if split.path == "/api/manual_edit":
                return self.send_json(apply_manual_edit(data))
            if split.path == "/api/prepare_frames":
                return self.send_json(start_prepare_frames(STATE.safe_video_path(data["path"])))
            if split.path == "/api/settings":
                return self.handle_settings(data)
            return self.send_error_json(404, "not found")
        except Exception as exc:
            self.send_error_json(500, f"{exc}\n{traceback.format_exc()}")

    def config_payload(self) -> dict[str, Any]:
        return {
            "video_root": str(STATE.video_root),
            "output_root": str(STATE.output_root),
            "views": list(VIEW_DIRS),
            "default_labels": LABELS,
            "gpu_id": STATE.gpu_id,
            "gpus": list_gpus(),
            "model_status": STATE.model_status,
            "model_error": STATE.model_error,
            "predictor_keys": sorted(STATE.predictors),
            "model_backend": STATE.model_backend,
            "checkpoint": str(STATE.checkpoint) if STATE.checkpoint else None,
            "default_checkpoints": {
                "sam21": str(DEFAULT_SAM21_CHECKPOINT),
                "sam31": str(DEFAULT_SAM31_CHECKPOINT),
            },
            "bpe_path": str(STATE.bpe_path) if STATE.bpe_path else None,
            "sam21_config": STATE.sam21_config,
            "inference_dtype": STATE.inference_dtype,
            "frame_extract_threads": STATE.frame_extract_threads,
        }

    def handle_settings(self, data: dict[str, Any]) -> None:
        if data.get("video_root"):
            root = resolve_video_root(Path(data["video_root"]), None)
            if not root.is_dir():
                raise FileNotFoundError(f"video root does not exist: {root}")
            STATE.video_root = root.resolve()
        if data.get("output_root"):
            STATE.output_root = Path(data["output_root"]).expanduser().resolve()
            STATE.output_root.mkdir(parents=True, exist_ok=True)
        gpu_id = data.get("gpu_id")
        gpu_id = None if gpu_id in (None, "") else int(gpu_id)
        if STATE.predictors and gpu_id != STATE.gpu_id:
            raise RuntimeError("GPU cannot be changed after SAM model is loaded; release model or restart server")
        STATE.gpu_id = gpu_id
        model_backend = data.get("model_backend")
        if model_backend:
            if model_backend not in {"sam21", "sam31"}:
                raise ValueError(f"invalid model backend: {model_backend}")
            if STATE.predictors and model_backend != STATE.model_backend:
                raise RuntimeError("model backend cannot be changed after SAM model is loaded; release model first")
            old_default = default_checkpoint_for_backend(STATE.model_backend)
            STATE.model_backend = model_backend
            if not data.get("checkpoint") and (STATE.checkpoint is None or STATE.checkpoint == old_default):
                STATE.checkpoint = default_checkpoint_for_backend(model_backend).resolve()
        if data.get("checkpoint"):
            checkpoint = Path(data["checkpoint"]).expanduser().resolve()
            if STATE.predictors and checkpoint != STATE.checkpoint:
                raise RuntimeError("checkpoint cannot be changed after SAM model is loaded; release model first")
            STATE.checkpoint = checkpoint
        if data.get("sam21_config"):
            sam21_config = str(data["sam21_config"]).strip()
            if STATE.predictors and sam21_config != STATE.sam21_config:
                raise RuntimeError("SAM2.1 config cannot be changed after SAM model is loaded; release model first")
            STATE.sam21_config = sam21_config
        if "bpe_path" in data:
            raw_bpe = str(data.get("bpe_path") or "").strip()
            bpe_path = Path(raw_bpe).expanduser().resolve() if raw_bpe else None
            if STATE.predictors and bpe_path != STATE.bpe_path:
                raise RuntimeError("BPE path cannot be changed after SAM model is loaded; release model first")
            STATE.bpe_path = bpe_path
        return self.send_json(self.config_payload())

    def handle_episodes(self) -> None:
        episodes: dict[str, dict[str, Any]] = {}
        for chunk in sorted(STATE.video_root.glob("chunk-*")):
            if not chunk.is_dir():
                continue
            for view, view_dir in VIEW_DIRS.items():
                for mp4 in sorted((chunk / view_dir).glob("episode_*.mp4")):
                    item = episodes.setdefault(mp4.stem, {"episode": mp4.stem, "chunk": chunk.name, "views": {}})
                    item["views"][view] = {
                        "path": str(mp4.resolve()),
                        "media_url": "/media?path=" + urllib.parse.quote(str(mp4.resolve())),
                    }
        for item in episodes.values():
            annotations = {}
            for view in VIEW_DIRS:
                out_dir = result_dir(item["episode"], view)
                files = {
                    "metadata": out_dir / "metadata.json",
                    "mask_int": out_dir / "mask_int.npy",
                    "object_masks": out_dir / "object_masks.npz",
                }
                annotations[view] = {
                    "done": all(path.exists() for path in files.values()),
                    "missing_files": [name for name, path in files.items() if not path.exists()],
                    "output_dir": str(out_dir),
                }
            item["annotations"] = annotations
            item["annotation_summary"] = {
                "done": sum(1 for status in annotations.values() if status["done"]),
                "total": len(VIEW_DIRS),
                "missing_views": [view for view, status in annotations.items() if not status["done"]],
            }
        return self.send_json({"episodes": sorted(episodes.values(), key=lambda x: x["episode"])})

    def send_static(self, rel: str) -> None:
        root = Path(__file__).resolve().parent / "static"
        rel = posixpath.normpath(urllib.parse.unquote(rel)).lstrip("/")
        path = (root / rel).resolve()
        if path != root and root not in path.parents:
            return self.send_error_json(403, "bad static path")
        return self.send_file(path)

    def send_file(self, path: Path, range_ok: bool = False) -> None:
        if not path.exists():
            self.send_error(404)
            return
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = HTTPStatus.OK
        if range_ok and "Range" in self.headers:
            unit, rng = self.headers["Range"].split("=", 1)
            if unit == "bytes":
                a, _, b = rng.partition("-")
                start = int(a) if a else 0
                end = min(int(b) if b else size - 1, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser-based multi-view video annotation with SAM2.1 and optional SAM3.1.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--output-root", "--result-root", dest="output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/vla_video_annotator"))
    parser.add_argument("--model-backend", choices=["sam21", "sam31"], default="sam21")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--sam21-config", default=DEFAULT_SAM21_CONFIG)
    parser.add_argument("--bpe-path", type=Path, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--inference-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--async-loading-frames", action="store_true", default=False)
    parser.add_argument("--frame-extract-threads", type=int, default=8)
    parser.add_argument("--preload-model", action="store_true")
    parser.add_argument("--no-preload-model", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    global STATE
    checkpoint = args.checkpoint or default_checkpoint_for_backend(args.model_backend)
    STATE = AppState(
        resolve_video_root(args.dataset_root, args.video_root),
        args.output_root,
        args.cache_root,
        checkpoint,
        args.bpe_path,
        args.model_backend,
        args.sam21_config,
        args.gpu_id,
        args.inference_dtype,
        args.async_loading_frames,
        args.frame_extract_threads,
    )
    print(f"SAM annotator ({STATE.model_backend}): http://{args.host}:{args.port}", flush=True)
    print(f"video root: {STATE.video_root}", flush=True)
    print(f"output root: {STATE.output_root}", flush=True)
    print(f"checkpoint: {STATE.checkpoint}", flush=True)
    print(f"frame extract threads: {STATE.frame_extract_threads}", flush=True)
    if args.preload_model and not args.no_preload_model:
        try:
            print(f"preloading {STATE.model_backend} model ...", flush=True)
            get_predictor()
            print(f"{STATE.model_backend} model loaded.", flush=True)
        except Exception:
            traceback.print_exc()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
