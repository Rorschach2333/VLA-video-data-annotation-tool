# VLA Video Annotation Tool

面向 VLA/机器人数据的网页式多视角视频分割标注工具。本项目 heavily based on Meta Segment Anything workflows，并将 SAM 系列模型作为外部推理后端。工具默认处理三个同步视角：`head`、`left`、`right`，输出 NumPy mask、bbox、prompt 元数据、可视化 overlay 和手动调整记录。

默认后端：**SAM2.1**，用于 point/box prompt。**SAM3.1** 作为可选后端，适合 text prompt 流程。

[English README](README.md)

## 界面示例

![Head 视角结果](docs/images/head-result.png)

![Left 视角结果](docs/images/left-result.png)

## 功能

- 网页内置中文/英文切换，默认中文。
- 三视角视频浏览与逐帧 prompt 定位。
- 支持点、框、文本、画笔、橡皮工具。
- 支持后台 episode 队列，可选多 GPU 并行处理多个视角。
- 支持结果预览、overlay 渲染和手动 mask 修正。
- 本地 Python 标准库 HTTP 服务，无需额外 Web 框架。

## 数据目录

默认读取以下结构：

```text
<dataset-root>/
  videos/
    chunk-000/
      observation.images.ros_head/episode_000000.mp4
      observation.images.ros_left/episode_000000.mp4
      observation.images.ros_right/episode_000000.mp4
```

启动参数可以传 `<dataset-root>`，也可以直接传 `videos/` 目录。

## 安装

环境要求：

- Python 3.10+
- `ffmpeg` 和 `ffprobe` 在 `PATH` 中可用
- 支持 CUDA 的 PyTorch 环境
- 已安装 SAM2.1，并保证 `from sam2.build_sam import build_sam2_video_predictor` 可执行

```bash
git clone <repo-url>
cd VLA-video-data-annotation-tool

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

请按你的 CUDA 版本安装 PyTorch，再按 Meta SAM/SAM2 官方说明安装对应依赖。

模型权重可放在 `checkpoints/`，也可以启动时显式指定：

```text
checkpoints/sam2.1_hiera_base_plus.pt
checkpoints/sam3.1_multiplex.pt     # 可选
```

## 启动

SAM2.1 point/box 标注流程：

```bash
vla-video-annotator \
  --dataset-root /path/to/dataset \
  --output-root /path/to/annotation_outputs \
  --model-backend sam21 \
  --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
  --host 127.0.0.1 \
  --port 7861
```

浏览器打开 `http://127.0.0.1:7861`。

可选 SAM3.1 text prompt 流程：

```bash
vla-video-annotator \
  --dataset-root /path/to/dataset \
  --output-root /path/to/annotation_outputs \
  --model-backend sam31 \
  --checkpoint /path/to/sam3.1_multiplex.pt
```

## 启动参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset-root` | 未设置 | 数据集根目录。如果目录下存在 `videos/`，自动使用该子目录。 |
| `--video-root` | `data/videos` | 直接指定 `videos/` 目录。 |
| `--output-root`, `--result-root` | `outputs/annotations` | 标注结果输出目录。 |
| `--cache-root` | `/tmp/vla_video_annotator` | 抽帧缓存目录。 |
| `--model-backend` | `sam21` | 可选 `sam21` 或 `sam31`。 |
| `--checkpoint` | `checkpoints/` 下的后端默认文件 | 模型 checkpoint 路径。 |
| `--sam21-config` | `configs/sam2.1/sam2.1_hiera_b+.yaml` | SAM2.1 配置名或配置路径。 |
| `--bpe-path` | 未设置 | 可选 SAM3.1 BPE/tokenizer 文件路径。 |
| `--gpu-id` | 未设置 | CUDA GPU ID。为空时使用进程当前 CUDA 默认设备。 |
| `--inference-dtype` | `bfloat16` | 可选 `float32` 或 `bfloat16`。 |
| `--async-loading-frames` | false | 后端支持时启用异步加载帧。 |
| `--frame-extract-threads` | `8` | FFmpeg 抽帧线程数。 |
| `--preload-model` | false | 服务启动时立即加载模型。 |
| `--host` | `127.0.0.1` | HTTP 绑定地址。 |
| `--port` | `7861` | HTTP 端口。 |

常用默认值也可以用环境变量指定：

```bash
export VLA_ANNOTATOR_VIDEO_ROOT=/path/to/dataset/videos
export VLA_ANNOTATOR_OUTPUT_ROOT=/path/to/annotation_outputs
export SAM21_CHECKPOINT=/path/to/sam2.1_hiera_base_plus.pt
export SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt
export SAM21_CONFIG=configs/sam2.1/sam2.1_hiera_b+.yaml
```

## 使用流程

1. 启动服务并打开网页。
2. 在左侧选择界面语言，默认中文。
3. 设置数据集位置、输出目录、GPU、后端和 checkpoint。
4. 选择 episode。
5. 分别拖动三个视角的帧滑条，到目标第一次出现的起始帧。
6. 设置类别，默认类别为 `manip_obj` 和 `target`。
7. SAM2.1 推荐使用点或框；如需 text prompt，可切换到 SAM3.1。
8. 点击 **预览起始帧** 检查 prompt。
9. 点击 **加入任务列表** 或 **提交三视角任务** 执行传播。
10. 在 Result 区域用画笔/橡皮修正后，点击 **保存手动调整**。

## 输出

```text
outputs/annotations/<episode>/<view>/
  mask_int.npy       # uint16 [T, H, W]，0 为背景，1..N 为类别 ID
  boxes_xywh.npy     # float32 [T, N, 4]，x/y/w/h，空帧为 NaN
  object_masks.npz   # 每个类别一个 bool mask，[T, H, W]
  metadata.json      # 类别、视频信息、统计信息、输出文件索引
  prompts.json       # text、point、box prompt
  overlays/*.png     # 结果 overlay
  manual_edits.jsonl # 手动调整记录
```

## 隐私说明

发布仓库不包含数据集、模型权重、生成标注、缓存文件或私人绝对路径。真实数据路径请通过启动参数、环境变量或本地配置传入，不要提交到仓库。

## 开源协议

本项目中原创的标注工具代码使用 Apache License 2.0。

本项目 heavily based on Meta Segment Anything 的概念、API 与模型工作流。Meta SAM 系列代码、模型权重、checkpoint 和相关资源属于独立依赖，受其各自许可证与使用条款约束。归属与第三方许可证说明见 `NOTICE`。
