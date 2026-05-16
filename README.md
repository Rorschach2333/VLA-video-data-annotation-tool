# VLA Video Annotation Tool

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Backend](https://img.shields.io/badge/backend-SAM2.1%20%7C%20SAM3.1-111827.svg)](NOTICE)
[![UI](https://img.shields.io/badge/UI-Chinese%20%7C%20English-2563eb.svg)](README.zh-CN.md)

Browser-based multi-view video segmentation annotation for VLA and robot datasets.

This project is **based heavily on Meta Segment Anything workflows**. It provides a focused annotation UI and local server around SAM-family video segmentation backends, without vendoring SAM code, model weights, datasets, or generated annotations.

Default workflow: **SAM2.1 + point/box prompts**.

Optional workflow: **SAM3.1 + text prompts**.

中文文档: [README.zh-CN.md](README.zh-CN.md)

![Web interface](docs/images/image-web.jpg)

## What It Does

| Area | Support |
| --- | --- |
| Views | Three synchronized camera views: `head`, `left`, `right` |
| Prompts | Point, box, text prompt |
| Manual editing | Brush, eraser, per-frame result correction |
| Queue | Episode queue with optional multi-GPU view parallelism |
| Output | NumPy masks, boxes, object masks, prompt metadata, overlays |
| UI language | Chinese by default, English selectable in the web page |

## Install

This tool expects a working Meta SAM environment. Set up the Python/CUDA environment by following the official **SAM3 installation process** first:

- Official SAM3 repository: <https://github.com/facebookresearch/sam3>
- Official SAM2 repository, needed for the default `sam21` backend: <https://github.com/facebookresearch/sam2>

After the SAM environment is working, install this annotation tool into the same environment:

```bash
git clone https://github.com/Rorschach2333/VLA-video-data-annotation-tool.git
cd VLA-video-data-annotation-tool
python -m pip install -e .
```

Runtime requirements outside Python:

- `ffmpeg`
- `ffprobe`
- CUDA GPU supported by your SAM installation

Model checkpoints are not included. Keep them outside git, for example:

```text
checkpoints/
  sam2.1_hiera_base_plus.pt
  sam3.1_multiplex.pt        # optional
```

## Quick Start

SAM2.1 point/box annotation:

```bash
vla-video-annotator \
  --dataset-root /path/to/dataset \
  --output-root /path/to/annotation_outputs \
  --model-backend sam21 \
  --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
  --host 127.0.0.1 \
  --port 7861
```

Open:

```text
http://127.0.0.1:7861
```

Optional SAM3.1 text-prompt annotation:

```bash
vla-video-annotator \
  --dataset-root /path/to/dataset \
  --output-root /path/to/annotation_outputs \
  --model-backend sam31 \
  --checkpoint /path/to/sam3.1_multiplex.pt
```

## Dataset Layout

Pass either the dataset root or the `videos/` directory directly.

```text
<dataset-root>/
  videos/
    chunk-000/
      observation.images.ros_head/
        episode_000000.mp4
      observation.images.ros_left/
        episode_000000.mp4
      observation.images.ros_right/
        episode_000000.mp4
```

## Workflow

1. Start the server and open the browser UI.
2. Choose language in the sidebar. Chinese is the default.
3. Set dataset path, output path, GPU, backend, and checkpoint.
4. Select an episode.
5. Move each view to the first frame where the target should be tracked.
6. Keep or edit the default labels: `manip_obj`, `target`.
7. Add point/box prompts for SAM2.1, or switch to SAM3.1 for text prompts.
8. Preview the start frame.
9. Add the episode to the queue or submit all views.
10. Correct result masks with brush/eraser and save manual edits.

## Output Format

```text
outputs/annotations/<episode>/<view>/
  mask_int.npy
  boxes_xywh.npy
  object_masks.npz
  metadata.json
  prompts.json
  overlays/*.png
  manual_edits.jsonl
```

| File | Meaning |
| --- | --- |
| `mask_int.npy` | `uint16 [T, H, W]`, `0` for background, `1..N` for label IDs |
| `boxes_xywh.npy` | `float32 [T, N, 4]`, `x/y/w/h`, `NaN` for empty frames |
| `object_masks.npz` | Boolean mask array per label |
| `metadata.json` | Video info, labels, stats, and file index |
| `prompts.json` | Text, point, and box prompts |
| `overlays/*.png` | Rendered preview/result overlays |
| `manual_edits.jsonl` | Manual correction records |

## Server Options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset-root` | unset | Dataset root. If it contains `videos/`, that child directory is used. |
| `--video-root` | `data/videos` | Direct path to the `videos/` directory. |
| `--output-root`, `--result-root` | `outputs/annotations` | Annotation output directory. |
| `--cache-root` | `/tmp/vla_video_annotator` | Extracted-frame cache. |
| `--model-backend` | `sam21` | `sam21` or `sam31`. |
| `--checkpoint` | backend default under `checkpoints/` | Model checkpoint path. |
| `--sam21-config` | `configs/sam2.1/sam2.1_hiera_b+.yaml` | SAM2.1 config name/path. |
| `--bpe-path` | unset | Optional SAM3.1 tokenizer/BPE path. |
| `--gpu-id` | unset | CUDA GPU ID. Empty means the process default. |
| `--inference-dtype` | `bfloat16` | `float32` or `bfloat16`. |
| `--async-loading-frames` | false | Enable backend async frame loading when supported. |
| `--frame-extract-threads` | `8` | FFmpeg extraction thread count. |
| `--preload-model` | false | Load model at server startup. |
| `--host` | `127.0.0.1` | HTTP bind host. |
| `--port` | `7861` | HTTP port. |

Common environment variables:

```bash
export VLA_ANNOTATOR_VIDEO_ROOT=/path/to/dataset/videos
export VLA_ANNOTATOR_OUTPUT_ROOT=/path/to/annotation_outputs
export SAM21_CHECKPOINT=/path/to/sam2.1_hiera_base_plus.pt
export SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt
export SAM21_CONFIG=configs/sam2.1/sam2.1_hiera_b+.yaml
```

## Privacy

The repository intentionally excludes datasets, model weights, generated annotation files, frame caches, and private absolute paths. Keep local paths in command-line arguments, environment variables, or local-only scripts.

## License And Attribution

The original annotation-tool code in this repository is released under the [Apache License 2.0](LICENSE).

This project is based heavily on Meta Segment Anything concepts, APIs, and model workflows. Meta SAM-family code, checkpoints, weights, and related assets are separate dependencies and remain governed by their own licenses and terms. See [NOTICE](NOTICE).
