# 共享天线、多尺度时间、分层人数模型

这是独立的新结构实验，不覆盖旧0.54模型。无旧权重、旧偏置或保护规则。

按顺序运行：

1. `train.py`：两个版本×两个种子×四个留房间，共16次训练。
2. `evaluate.py`：比较同一880窗，选择版本与最终固定轮次。
3. `train.py --full`：所选版本两个种子用全部48条记录训练。
4. `export_model.py`：导出并验证真实输入数值。
5. `tools/check_candidate.py D:/huawei2026-experiments/shared_hierarchy_20261025/export`：独立CPU资源核验。
6. `export_model.py --publish` 与 `report.py`：保存模型及报告。

使用项目.venv的Python。权重与中间数据在D:/huawei2026-experiments/shared_hierarchy_20261025/。
已存在result.json的完整训练会被复用；修改配方须使用新的实验目录，不能混用旧结果。

所有校验按整记录隔离。历史留房间集合已用于开发，不是新盲测；来源960窗正确率不当作泛化准确率。平台0.54不能与本地准确率直接比较。

最终交付调整：双种子等权融合在开发集严重下降，保留该失败结果；按训练侧平均验证NLL选择seed2026单模型导出。两个全数据训练权重均保留。
