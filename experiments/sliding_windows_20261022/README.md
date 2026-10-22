# 滑动窗口数据复现

本轮暂停结构探索，沿用已测cross_events/context_events/mixer，从连续有标签CSI生成重叠600帧窗口。先读PLAN.md和../../docs/SLIDING_WINDOWS_20261022.md。

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/sliding_windows_20261022/prepare.py
.\.venv\Scripts\python.exe -X utf8 experiments/sliding_windows_20261022/audit.py
.\.venv\Scripts\python.exe -X utf8 experiments/sliding_windows_20261022/make_splits.py
.\.venv\Scripts\python.exe -X utf8 experiments/sliding_windows_20261022/train.py
.\.venv\Scripts\python.exe -X utf8 experiments/sliding_windows_20261022/audit_training.py
.\.venv\Scripts\python.exe -X utf8 experiments/sliding_windows_20261022/deliver.py
.\.venv\Scripts\python.exe -X utf8 experiments/sliding_windows_20261022/finalize.py
```

D:/huawei2026-experiments/sliding_windows_20261022/cache为按原记录分片的npy，只读内存映射训练；窗口CSV和JSON索引保留。旧锚点逐位复用，新增窗口使用高精度预处理，原始标签核对、缓存哈希与隔离审计独立运行。原48记录/960窗，扩展1秒与0.5秒起窗，仍仅48独立记录。

训练33次固定预算实验。`samples_per_epoch`匹配旧窗口数、`training_windows`是可选池大小、`actual_windows`是实际抽到的不同窗口数，三者不可混用。按类别/房间/源记录平衡，每源记录贡献相近。主要验证维持旧非重叠窗；dense_validation仅附加诊断，不可因窗口数量更多称置信度更高。训练与验证源记录ID不交叉。

缓存与metrics可恢复，修改预处理/网络/训练协议时必须另开目录；不要复用旧分片或已有拟合。三份ONNX和独立CPU检查完成后才交付。平台分数与本地分数分别登记，不覆盖原0.39/0.33模型。
