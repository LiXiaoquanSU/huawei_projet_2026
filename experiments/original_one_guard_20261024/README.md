# 冻结权重的原视角1人保护

这是基于实际0.47错误审计的单一决策规则消融，不是重训或新类别偏置。

从49130原文件复制全部图与权重，只增加三个节点：原始044分支argmax为1时返回其分数，否则返回旧49130分数。这可能覆盖旧0/2保护，代价需与收益一起报告。

执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/original_one_guard_20261024/build.py
.\.venv\Scripts\python.exe -X utf8 experiments/original_one_guard_20261024/report.py
```

实验、完整原始窗口数值检查及运行检查保存到 `D:/huawei2026-experiments/original_one_guard_20261024`。build在全部检查通过后才复制提交候选到 `models/original_one_guard_20261024/`。原模型不修改，平台分数字段保持未知。

本地规则来自看过训练数据后的错误分析，不是独立开发验证。旧的逐类不退步门槛属于上一轮新权重融合筛选；本轮明确是一次有两处空房退步的探索性推理消融，不声称通过该门槛或已改善隐藏集。
