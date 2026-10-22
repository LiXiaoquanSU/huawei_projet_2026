# Submission 47547：文件下载失败复查

用户提供的原文：`Submission 47547 could not be evaluated. Unable to download the submission file.` 上传的是 `submit_random_restart_best_20261020.onnx`。

这说明评测方未能取得文件，不能据此认定模型推理不兼容或准确率失败。当前登记为下载失败、成绩未知；没有修改网络、重新训练或赋零分。

## 本地证据

- 根目录文件、models/random_restart_20261020/01_wide_seed42.onnx和D盘原导出文件三者SHA256一致：`3a2ba0de05dc81a726f7c06be71a748a6ac6b414c9b2d2bcd51f88745ec3f7ed`。
- 97,913,568字节，小于100,000,000字节；ONNX检查通过，权重完整内嵌，不依赖额外文件。
- 本轮重新运行src/check_model.py通过，float32官方输入/输出形状正确、输出有限，CPU平均约0.065秒。之前三种输入的完整资源/数值检查也通过。
- 与用户已报告平台0.33的上一份97.91MB模型比较，IR版本、opset、算子序列、参数名/类型/形状相同。二者训练权重不同；这仅支持兼容性对照，不能证明远端链接可访问。
- 无评测方附件URL或后台日志，不能判定上传未完成、存储URL授权/过期或后台网络中的具体原因。

## 重传文件

已生成项目 `submission_retry_47547/model.onnx`，即：

`C:/Users/lxq_1/Desktop/huawei2026/submission_retry_47547/model.onnx`

只改路径/文件名，字节与哈希完全相同。根目录原 `model.onnx` 仍是0.35文件，没有覆盖。标准短名用于重新建立附件，不代表已经找到文件名是故障原因，也不能保证修复平台存储问题。

根据[本地官方提交指南](reference/submission_guide.pdf)，在Future Network (Data Communications)的模型提交字段重新选择该文件；等待上传完成，确认附件在表单中显示。上传会自动触发评测，无须点击Submit your participation。两次提交至少间隔30分钟，每24小时最多20次。

若再次出现相同下载错误，应通过队伍admin chat请主办方检查附件和评测下载日志。可以复制以下说明自行发送（本次未替用户发送）：

> Submission 47547 failed with "Unable to download the submission file." The original uploaded filename was submit_random_restart_best_20261020.onnx (97,913,568 bytes). The local ONNX file passes validation and CPU inference and contains embedded weights. Could you check the attachment's storage URL and the evaluator's download/access logs? SHA256: 3a2ba0de05dc81a726f7c06be71a748a6ac6b414c9b2d2bcd51f88745ec3f7ed.

复查证据：checks/download_47547/report.json。该复查不证明老师平台已恢复，需新提交结果确认。

## 重传后的用户反馈

用户报告标准名副本仍无法通过，但未提供新的报错原文或提交编号。不能确认是下载请求超时、URL失效、提交窗口或推理限制。本次不继续重复上传大号，也不重训；当时改试原宽度随机重训seed42（02_original_seed42.onnx，78.61MB；现已获得0.37），标准名副本在submissions/next_original_seed42/model.onnx。复查脚本已归档，原报告保留。

## 原宽度seed42平台0.37反馈（2026-10-20）

用户报告本次原宽度随机重训seed42取得0.37。正式文件为 `models/random_restart_20261020/02_original_seed42.onnx`，标准名副本为 `submissions/next_original_seed42/model.onnx`；SHA256均为 `c4fa83755c11a057e970649004ad0fb4dd582dc98fb7acce569ec28714270e74`，大小78,608,334字节。提交编号未提供。

它与历史82.06MB的0.37版是不同权重，现并列最佳；本轮随机初始化重训复现了最高分，但尚未提高最高分。原四路结构仍值得保留为基准，此结果不能证明结构已到上限，也不能证明seed42在其他训练设置下总是最好。原宽度开发房间均值28.1%与此次平台37%来自不同数据与评估流程，分别保留，不能互相替换。

大号seed42仍是下载失败、未评测；原宽度提交成功不能确定大号失败原因。已测扩宽微调版0.33也不是这个大号随机初始化文件。本次仅更新成绩、索引与经验总结，保留核心代码、所有模型与D盘原始实验记录，没有新增训练。
