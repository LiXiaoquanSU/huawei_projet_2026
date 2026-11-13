# 环境适应与天线图双验证

报告：`docs/ENVIRONMENT_PROTOCOL_20261025.md`。大文件保存在
`D:/huawei2026-experiments/environment_protocol_20261025/`，不放在C盘仓库。

依赖项目已有的实际ONNX预处理缓存、四份历史留房间教师、四路线证据缓存；不是独立数据包。
`run.py`、`graph.py`中的路径给出了确切依赖。训练不读取老师隐藏测试标签。

从项目根目录使用项目虚拟环境依次执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/environment_protocol_20261025/run.py
.\.venv\Scripts\python.exe -X utf8 experiments/environment_protocol_20261025/audit_data.py
.\.venv\Scripts\python.exe -X utf8 experiments/environment_protocol_20261025/evaluate.py
.\.venv\Scripts\python.exe -X utf8 experiments/environment_protocol_20261025/subspace.py
.\.venv\Scripts\python.exe -X utf8 experiments/environment_protocol_20261025/train_graph.py
.\.venv\Scripts\python.exe -X utf8 experiments/environment_protocol_20261025/evaluate_graph.py
.\.venv\Scripts\python.exe -X utf8 experiments/environment_protocol_20261025/report.py
```

已有训练权重会复用；需要改变训练配方时应新建实验目录，避免混用旧缓存。
必须完成各阶段全部训练再运行对应评估。步骤/融合比例只由源校准集选择，目标人数标签只用于训练后的评估。
`report.py`核验图分支抽样哈希、记录互斥、Rx置换不变性，并汇总三阶段报告；单独重跑`evaluate.py`后需要再运行`report.py`恢复完整报告。

主要审计产物：`protocols.json`、仓库`docs/ENVIRONMENT_PROTOCOL_SUPPORT_20261025.csv`、
`data_audit.json`、`fresh_ati_audit.csv`、`selection.json`、`subspace_results.json`、
`graph/selection.json`与`graph/fit_audit.json`。

本轮不自动提交平台或推送Git。未达到跨房间筛选门槛的权重只作研究记录，不列入推荐提交模型。
