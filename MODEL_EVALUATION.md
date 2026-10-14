# 模型来源、许可与兼容性核实

Matting 来源与许可核实日期：2026-09-23；DIS5K 本地运行核实日期：2026-10-13。

## 当前默认基线：DIS5K

| 项目 | 结论 |
| --- | --- |
| 模型 | `onnx-community/BiRefNet-DIS5K-ONNX` 的 `onnx/model.onnx`，本地文件 `models/birefnet-dis5k.onnx` |
| SHA-256 | `6470117bac6f8d82a3f62921056f52d0f5c4d36d1d832096331d5ea38a03acb5` |
| ONNX 输入 | `input_image`，`[1, 3, 1024, 1024]` float32，沿用现有 ImageNet mean/std 预处理 |
| ONNX 输出 | `output_image`，`[1, 1, 1024, 1024]` float32 logits；经 sigmoid 得到软 Alpha，再裁去补边并恢复原图尺寸 |
| 运行时 | ONNX Runtime 1.19.2，`CPUExecutionProvider` 本机创建会话与真实推理通过 |
| 效果边界 | 锅具、艺术字已有同参数四图对照；用户确认整体优于 Matting，但正式业务验收、透明/半透明材质和 Windows 真机仍待验证 |
| 许可边界 | 本次默认切换未重新核实 DIS5K 仓库及上游许可证文本；商业发布前须完成许可证、署名及供应链复核 |

[DIS5K ONNX 模型仓库](https://huggingface.co/onnx-community/BiRefNet-DIS5K-ONNX)；本地对照证据见 `outputs/compare-matting-dis5k-20261013-120611/`，不随源码仓库上传。

## 可选旧基线：Matting

| 项目 | 结论 |
| --- | --- |
| 模型 | `ZhengPeng7/BiRefNet-matting` 的 ONNX 格式转换 `emrikol/birefnet-matting-onnx` |
| 任务 | 通用 trimap-free matting，输出 alpha matte |
| ONNX 输入 | `[1, 3, 1024, 1024]` float32，ImageNet mean/std |
| ONNX 输出 | `[1, 1, 1024, 1024]` logits，sigmoid 后作为 alpha |
| 运行时 | `onnxruntime` CPU，Windows/macOS 可用，不要求 CUDA |
| 许可 | ONNX 仓库标注 MIT，并说明继承上游 BiRefNet 许可与署名要求 |
| 体积 | 模型卡显示约 941 MB，fp32 |
| 风险 | ONNX 文件是第三方格式转换，不等于上游官方发布；进入商业发布前需复核许可证文本、署名和模型供应链固定哈希 |

来源链接：

- [上游 BiRefNet-matting 模型卡](https://huggingface.co/ZhengPeng7/BiRefNet-matting)
- [ONNX 格式转换模型卡](https://huggingface.co/emrikol/birefnet-matting-onnx)
- [BiRefNet 官方代码仓库](https://github.com/ZhengPeng7/BiRefNet)

## 排除的候选

`briaai/RMBG-2.0` 的模型卡说明其输出为 8-bit alpha matte，能力方向匹配；但当前权重使用 gated 的 CC BY-NC 4.0，商业使用需要与 BRIA 另行签约。因此本项目不把它作为企业工具默认模型。

## 兼容性结论

ONNX Runtime 官方文档建议 macOS/Arm 使用 CPU 包，并列出 Windows 与 macOS CPU 支持。项目统一使用 `onnxruntime` CPU 包，避免 CUDA 和 NVIDIA 依赖；后续可在不改变推理接口的情况下增加平台可选加速。

兼容性参考：[ONNX Runtime Python 安装文档](https://onnxruntime.ai/docs/get-started/with-python.html)。
