# 模型文件索引

最终推荐文件是仓库根目录的[model.onnx](../model.onnx)，原版G′，已报告公开平台0.54。文件大小和SHA见[final_submission.json](final_submission.json)。

[registry.json](registry.json)保存历史模型身份与实验状态；[platform_scores.csv](platform_scores.csv)保存平台反馈。历史研究文件不因为本地分数高就视为新的提交推荐。

其他模型、训练权重和重复提交副本保留本地并由.gitignore排除。此前已纳入Git的best_040/model.onnx保留作为历史版本，勿与最终文件混淆。
