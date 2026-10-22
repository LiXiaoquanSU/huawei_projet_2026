# WiFi CSI 人数识别

华为 France Tech Arena 2026 项目：从 WiFi CSI 的 600 帧窗口预测 0–4 人。

## 当前结果

| 模型 | 平台分数 | 本地文件 |
|---|---:|---|
| 跨注意力事件，seed42 | **0.39** | `models/best_039/model.onnx` |
| 上下文事件，seed2026 | **0.39** | `models/best_039_context/model.onnx` |
| 局部事件 | 0.38 | `models/best_038/model.onnx` |
| 四路线原版 | 0.37 | `models/best_037/model.onnx` |
| 原始小型 Mixer | 0.33 | `models/fresh_backbones_20261022/01_mixer_seed42.onnx` |
| 1 秒滑窗 Mixer | 0.28 | `models/sliding_windows_20261022/03_mixer_step_1s_seed42.onnx` |

下一份待测是 `models/sliding_windows_20261022/01_context_events_step_0p5s_seed2026.onnx`（70.78 MB）。这是重新训练的权重，平台成绩未知，不继承原版 0.39。根目录 `model.onnx` 是旧 0.35 模型。

完整成绩与文件 SHA256 见 [成绩表](models/platform_scores.csv) 和 [模型登记](models/registry.json)。模型权重、原始数据及生成结果只保存在本机，不上传 GitHub。

## 代码位置

- `src/`：基础网络、CSI 预处理、损失、训练、ONNX 导出与检查。
- `experiments/structure_events_20261021/`：局部事件结构和参考训练流程。
- `experiments/global_local_20261021/`：两份 0.39 模型的网络与训练。
- `experiments/sliding_windows_20261022/`：按实际时间生成重叠窗口、按记录划分、训练和审计。
- 其他 `experiments/`：历史结构与数据对照，保留用于复现，不代表当前推荐。
- `tools/check_candidate.py`：Windows 上隔离进程的 ONNX CPU 数值、速度、内存检查。
- `docs/`：研究报告、经验与实验索引。

## 环境与运行

Windows、Python 虚拟环境；深度训练使用 CUDA，提交模型使用 CPU ONNX Runtime。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/train.py --help
.\.venv\Scripts\python.exe src/export_model.py --help
.\.venv\Scripts\python.exe src/check_model.py --help
```

实验脚本是本机研究流程，仍包含 D 盘数据路径和已有缓存依赖。迁移电脑时先检查各脚本中的 `WORK`、`RAW`、`OLD` 等路径；不能在仅克隆代码后直接复现已有训练结果。原始比赛数据位于 `D:/huawei2026-data/labeled`，实验权重、缓存和逐次指标位于 `D:/huawei2026-experiments`。部分历史导出还需要本机的 `checks/smoke_input.npz` 与归档中的异常间隔测试样本，这些测试输入未随仓库分发。

## 数据与验证原则

48 条有标签记录，已知房间仅 5 个，人数与房间分布偏斜。原 960 个非重叠窗口扩展为 1 秒起窗 3780 个、0.5 秒起窗 6637 个；独立记录仍然只有 48 条。

必须先按源记录/房间分开训练和验证，再切窗口。同一记录的重叠帧不能跨训练与验证。主要验证保持原非重叠窗口，密集窗口仅作额外诊断。本地留房间准确率与老师平台准确率不是同一个指标。

提交接口：`frames` float32 `[600,1,3,1024,2]`，`intervals` float32 `[600]`，输出 `logits` float32 `[5]`；ONNX 内置权重。项目本地检查采用文件小于 100,000,000 字节、CPU 均值小于 0.7 秒、峰值内存小于 6 GB 的约束；本机通过不保证平台成绩或下载成功。

[经验总结](docs/SUMMARY.md) · [实验索引](docs/EXPERIMENT_INDEX.md) · [重叠窗口报告](docs/SLIDING_WINDOWS_20261022.md) · [整理记录](docs/CLEANUP_20261022.json)
