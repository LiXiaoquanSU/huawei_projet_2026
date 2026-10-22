# 局部事件与多时延关系结构实验

后续平台反馈：events seed42已获0.38，当前最佳。只用比赛有标签数据，从头训练。最佳副本models/best_038/model.onnx；见docs/MODEL_038.md及checks/events_038_feedback.json。以下流程保留生成时记录，不应重跑deliver.py/finalize.py覆盖新成绩；这两个入口已加保护。新训练用新的实验目录。

官方数据、固定原预处理和原训练预算下，只改结构。完整假设与预设选择规则见PLAN.md，网络见network.py。数据、完整权重、预测、训练日志和源码快照在D:/huawei2026-experiments/structure_events_20261021。

顺序运行feedback.py、train.py、residual.py、deliver.py、finalize.py，使用项目.venv Python。train.py会跳过已完成的同名拟合；重跑不覆盖已保存指标。改变代码/计划需要新实验目录，不能复用旧结果冒充新结果。初始events的e1压力测试全错，因此按RESIDUAL_AMENDMENT.md只追加一次保留原注意力的门控事件修订。finalize.py仅针对本轮已确认的events导出，登记已知失败，不通用于其他结构。

feedback.py登记前一轮精确SHA文件的平台0.35，不给未测对照记分。train.py先四方案三房间筛选，再两种子复验和e1压力测试，最后全量48记录训练一个新候选。deliver.py负责ONNX导出、独立CPU检查、旧模型哈希、训练记录隔离、新参数实际更新核验，以及报告/成绩表登记。

新提交副本固定为submissions/structure_events_20261021/model.onnx，具体启用结构、种子和哈希见同目录metadata.json。平台结果未知；不要重跑上一轮report.py覆盖后续0.35反馈。原0.37模型与提交入口不替换。
