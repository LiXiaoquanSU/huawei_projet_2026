# 模型记录

当前平台并列最佳：`best_039/model.onnx`（跨注意力）和 `best_039_context/model.onnx`（上下文），均为 0.39。

`platform_scores.csv` 按具体文件 SHA256 记录平台反馈；空白分数表示未知。`registry.json` 保存更多训练和来源信息。权重仅在本机保留，GitHub 不包含 ONNX 或训练检查点。

最近滑窗 Mixer 为 0.28；下一份待测是 `sliding_windows_20261022/01_context_events_step_0p5s_seed2026.onnx`。原结构分数不能转移到新权重。
