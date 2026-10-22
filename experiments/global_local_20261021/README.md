# 全局关系与局部事件

后续平台反馈：跨注意力seed42已获0.39，当前最佳副本models/best_039/model.onnx；上下文事件seed2026尚未测试、分数留空。记录见record_039.py、checks/cross_events_039_feedback.json及docs/MODEL_039.md。不要重跑原导出流程覆盖已取得的反馈；入口已加保护。以下保留生成时流程。

先读PLAN.md。新结构在network.py；train.py引用已验证的0.38原训练循环，但输出全部重定向到新的D:/huawei2026-experiments/global_local_20261021，不修改旧权重或成绩。正式流程为项目.venv Python运行train.py，再运行deliver.py。

repair_axis.py是已经执行的一次启动修复工具，不应在正式训练完成后重跑（会归档fits）。_invalid_segment_axis_v0保留错误轴版本，不参与正式结果。新统计按同片段特征拼接，三结构已通过片段身份/形状/梯度核验。

baseline/events控制复用旧折外记录，seed42a1重新预测核对一致。只用比赛有标签数据、无外部/无U/无预训练，不从完整0.38权重初始化开发模型。19次新拟合，最多两个新研究候选，文件身份和平台未知状态以models/global_local_20261021及submissions/global_local_20261021的JSON为准。旧0.38仍为平台已评测最佳。
