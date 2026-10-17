# 跨平台精细抠图工具

本项目是独立开发中的“精细抠图”后端与浏览器开发预览。用户可在本机上传原图、查看任务状态并下载透明 PNG。正式业务系统接入尚未完成。

当前默认使用 BiRefNet-DIS5K ONNX；原 BiRefNet-matting ONNX 保留为可选模型。两者均由 ONNX Runtime CPU 执行，Windows 与 macOS 共用同一套源码。模型来源、许可证与限制见 [MODEL_EVALUATION.md](MODEL_EVALUATION.md)。

## 环境

- macOS 或 Windows
- 64 位 Python 3.9+
- 仅用默认 DIS5K 建议至少 2 GB 可用磁盘；若同时保留 Matting，建议至少 3 GB（两个模型分别约 973 MB、941 MB）
- 不需要 NVIDIA/CUDA

## 安装与下载模型

macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_model.py --url 'https://huggingface.co/onnx-community/BiRefNet-DIS5K-ONNX/resolve/main/onnx/model.onnx' --destination models/birefnet-dis5k.onnx
shasum -a 256 models/birefnet-dis5k.onnx
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\download_model.py --url 'https://huggingface.co/onnx-community/BiRefNet-DIS5K-ONNX/resolve/main/onnx/model.onnx' --destination models\birefnet-dis5k.onnx
(Get-FileHash models\birefnet-dis5k.onnx -Algorithm SHA256).Hash
```

两种系统的哈希结果都应为 `6470117bac6f8d82a3f62921056f52d0f5c4d36d1d832096331d5ea38a03acb5`；不一致时不要启动推理。已有正确文件时，下载脚本只显示现有文件哈希，不会覆盖。若需单独下载可选 Matting，运行不带参数的 `python scripts/download_model.py`（Windows 为 `python scripts\download_model.py`）。

## 启动开发预览

必须显式提供开发 token 和开发 owner；不会使用默认测试用户，也不会接受浏览器传入的 `user_id`。服务只允许绑定本机回环地址。

macOS：

```bash
export CUTOUT_DEV_TOKEN='仅用于本机预览的随机字符串'
export CUTOUT_DEV_OWNER='local-owner'
PYTHONPATH=src python -m cutout_tool.server \
  --port 8765 \
  --data-dir /private/tmp/aigc-cutout-data
```

Windows PowerShell：

```powershell
$env:CUTOUT_DEV_TOKEN = '仅用于本机预览的随机字符串'
$env:CUTOUT_DEV_OWNER = 'local-owner'
$env:PYTHONPATH = 'src'
python -m cutout_tool.server `
  --port 8765 `
  --data-dir "$env:TEMP\aigc-cutout-data"
```

打开 `http://127.0.0.1:8765/`，在页面输入同一个 token。`/private/tmp` 或 Windows 临时目录用于绕开 NAS/SMB 对 SQLite 事务文件的限制；正式后端应使用本机或容器本地持久化磁盘。

页面会提供：拖拽上传、任务状态、原图/结果对照、缩放、透明/黑底/白底/Alpha 预览和透明 PNG 下载。

## API（当前开发预览契约）

所有 `/api/*` 请求都必须带 `X-Cutout-Dev-Token`。正式接入时，这一层要替换为业务系统的服务端身份适配器，不能把开发 token 带入生产。

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查与模型存在性 |
| POST | `/api/tasks` | `multipart/form-data` 的 `file` 字段创建任务，返回 202 |
| GET | `/api/tasks` | 当前开发 owner 的历史任务 |
| GET | `/api/tasks/{task_id}` | 当前 owner 的任务状态和结果元数据 |
| POST | `/api/tasks/{task_id}/retry` | 失败任务有限重试一次 |
| GET | `/api/tasks/{task_id}/files/{kind}` | `original`、`transparent`、`black`、`white`、`alpha` |

状态为 `queued`、`processing`、`succeeded`、`failed`；服务重启时未完成任务会变为失败并给出可重试提示，不会长期卡在 processing。上传限制为 25 MB、40 MP，文件内容会重新校验格式，存储名称使用随机任务 ID。

幂等与并发控制（2026-10-17 升级）：

- **幂等去重**：上传按「owner + 图片内容 SHA-256」查重。重复提交同一张图直接返回已有任务，响应带 `reused: true`（成功任务返回 200），不重复推理；此前失败的重复图片复用同一任务 ID 并消耗其唯一一次重试。
- **单用户并发上限**：`MAX_INFLIGHT_PER_OWNER = 1`，同一 owner 已存在 `queued/processing` 任务时，新提交与重试均返回 409；不同 owner 互不影响，全局仍为单 worker 串行执行。
- **产物一致性**：任务标记 `succeeded` 前校验透明 PNG、黑/白底预览、Alpha 四件套均存在且非空（`infer.verify_outputs`），杜绝“数据库显示成功但文件缺失”。
- **旧库自动迁移**：启动时若任务表缺少 `image_sha256` 列则自动 `ALTER TABLE` 补充，并创建 `(owner_id, image_sha256)` 索引；历史任务哈希为空，不参与幂等匹配。

## CLI 真实推理

```bash
PYTHONPATH=src python -m cutout_tool \
  --input inputs/example.jpg \
  --output-dir outputs/example
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = 'src'
python -m cutout_tool `
  --input inputs\example.jpg `
  --output-dir outputs\example
```

以上默认使用 DIS5K。手动切回 Matting：CLI 命令追加 `--model models/birefnet-matting.onnx`（PowerShell 为 `--model models\birefnet-matting.onnx`）；开发预览服务启动命令也可追加相同的 `--model`。已在运行的服务不会自动切换，需按原启动方式重新启动后才生效。

每次真实推理生成透明 RGBA PNG、黑底预览、白底预览、Alpha 图和 `run.json`；原图尺寸会保留，已有透明区域不会被重新变成不透明。
JPEG 原图按 RGB 直接送入模型；黑底、白底只在生成 Alpha 后用于结果检查，不参与 JPEG 推理。四角度锅具边缘复核页仅保存在本地 `outputs/pan-edge-refined-20261012/review.html`，不随源码仓库上传。

## 已验证素材

用户提供的两张拼图已拆分为“原图输入”和“通过图参考”，原始 Downloads 文件保持不变。以下素材仅保存在本地，不随源码仓库上传：

- 锅具输入：`inputs/target_pan_original.png`；通过图参考：`references/target_pan_passed_panel.png`
- 艺术字输入：`inputs/target_gold_text_original.png`；通过图参考：`references/target_gold_text_passed_panel.png`

模型输出位于 `outputs/target_pan/` 与 `outputs/target_gold_text/`。这些是实际模型结果，当前只完成工程与效果核验，不宣称已经达到最终业务验收质量。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/private/tmp/aigc-cutout-pycache python -m compileall -q src tests
```

当前已完成 CLI/模型/API 自动化验证；开发预览页面尚未完成真实浏览器点击联调。正式业务系统的身份、登录态、权限、文件所有权与部署均未接入。本仓库仅供本地开发预览，不应将开发服务直接暴露到公网。
