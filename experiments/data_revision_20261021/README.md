# 0.34反馈后的数据修订训练

本轮先通过feedback.py按确切SHA登记前一版平台0.34，原0.37保留。不要重跑前一轮report.py覆盖该后续反馈。

所有数据、清洗缓存、完整权重、逐窗预测和资源结果保存在D:/huawei2026-experiments/data_revision_20261021。src和原模型不改。

按顺序：acquire.py读取作者Figshare索引与摘要；select_data.py使用bundled Python只读解析摘要（不改XLSX）；acquire.py --selected <selection.json>选择性下载；supplement.py只处理已登记的质量替补；prepare_dynamic.py核对真实标签/实测时间、首尾裁剪/断包排除、抗混叠/物理128频组；train.py --stage control；finish.py依次screen/replicate/full/export；verify.py独立复核；report.py整理结果。

详细超参数、外部Office1隔离、官方新增e1压力测试、粗频组修订说明见PLAN.md。_coarse_frequency_v1仅保留已停用的初始32频组试验，不参与新方案选择。正式80MHz输入保持128真实邻频聚合组；20MHzRadioPoints保持真实30频点，不造高分辨率数据。

当前外部数据不再混用Brescia名义时间与SHARP活动标签。新数据为EHUNAM CC BY4.0真实多人计数；混合变体才加RadioPoints。不存在把多接收机当相干天线、同次采集不同接收机/相邻窗跨训练验证的问题。

新候选固定副本为submissions/data_revision_20261021/model.onnx，身份以同目录metadata.json为准；前一版0.34仍在submissions/external_upgrade_20261021/model.onnx，两者不能混淆。原0.37最佳不因本地训练结果自动改排名。
