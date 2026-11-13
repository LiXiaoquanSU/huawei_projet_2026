# 固定推理策略研究

在0.47经验上比较7个固定规则，选出取消硬保护的完整12视角平均；冻结原权重和偏置，不新增训练。实验数据和指标说明见 [报告](../../docs/TTA_POLICY_20261024.md)。用户实测平台0.41，低于保护版0.47，不再推荐。后续校准迁移诊断见 [复盘](../../docs/REVIEW_041.md)。

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/tta_policy_20261024/evaluate.py
.\.venv\Scripts\python.exe -X utf8 experiments/tta_policy_20261024/deliver.py
.\.venv\Scripts\python.exe -X utf8 experiments/tta_policy_20261024/verify_export.py
.\.venv\Scripts\python.exe -X utf8 experiments/tta_policy_20261024/verify_symmetry.py
```

evaluate复用既有预测缓存。deliver禁止覆盖已有候选，重跑需选择新输出位置或保留已有结果。需要D盘历史checkpoint、原始CSI和本机保留的0.37检查点及0.44/0.47文件。结果记录在 D:/huawei2026-experiments/tta_policy_20261024。
