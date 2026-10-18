# Huawei 2026 · CSI 人数识别

使用 CSI 与 ATI 数据进行人数分类，包含数据审计、房间隔离验证、GPU 训练及 CPU ONNX 导出。

当前方案融合 1D 时序网络、时频 CNN 与 MixStyle 时频 CNN。开发阶段三个留出房间的宏平均准确率为 **42.69%**，窗口加权准确率为 **45.12%**。这些是反复选型使用的开发成绩，不是独立测试或比赛成绩；全量训练提交候选没有独立本地准确率。详见 [迭代报告](docs/ITERATION_REPORT.md)。

## 目录

| 位置 | 内容 |
| --- | --- |
| 根目录 `*_model.py`、`*_models.py` | 模型、预处理及公共依赖 |
| `train_baseline.py`、`run_iterations.py` | 基线训练及房间隔离实验 |
| `train_submission.py`、`export_improved.py` | 全量候选训练及开发模型导出 |
| `validate_model.py`、`verify_gpu_export.py` | 接口、数值及 CPU/GPU 验证 |
| `analyze_data.py`、`prepare_data.py` | 数据审计、划分及解压校验 |
| `cache_tf.py`、`compare_ensembles.py`、`summarize_improvements.py` | 特征缓存、融合比较及报告生成 |
| `experiments/` | 相位缓存、物理特征及投影实验入口 |
| `analysis/` | 数据审计报告、统计及固定划分清单 |
| `docs/` | 环境、数据策略、模型设计及训练记录 |
| `artifacts/` | 本地缓存、权重、预测及 ONNX；不上传 Git |

## 环境与数据

本项目原验证环境为 Windows、Python 3.13、PyTorch 2.9.1 CUDA 12.6。训练脚本需要 NVIDIA GPU；导出模型使用 ONNX Runtime CPU 推理。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

`requirements-lock.txt` 记录原机器环境的精确版本。详细环境记录见 [ENVIRONMENT.md](docs/ENVIRONMENT.md)。

按 [数据下载说明](docs/data_downloads.md) 获取数据，默认放在 `D:\huawei2026-data`。当前部分脚本与已保存清单包含该绝对路径；换机器时需修改脚本中的数据路径，并重新生成审计及划分清单：

```powershell
.\.venv\Scripts\python.exe -X utf8 analyze_data.py --data-root D:\huawei2026-data
```

仓库中的 `analysis/` 保留实验追溯记录；训练前应确保清单路径与本机数据一致。

## 运行顺序

请从仓库根目录运行所有命令。完整流程及历史参数见 [基线训练](docs/TRAINING.md) 与 [迭代报告](docs/ITERATION_REPORT.md)。

1. 准备环境、下载数据，运行 `analyze_data.py` 生成本机清单。
2. 运行 `train_baseline.py` 生成基线权重、特征缓存与训练/验证窗口清单。
3. 运行 `cache_tf.py` 生成时频缓存，再用 `run_iterations.py` 运行所需开发实验（参数见 `--help`）。
4. 用 `compare_ensembles.py` 比较融合，`export_improved.py` 导出选定开发模型。
5. `train_submission.py` 依赖选中组件的开发实验指标、基线和时频缓存，以及 `artifacts/model_design/` 的验证用例；后者由 `validate_model.py` 的默认流程生成。准备齐全后才运行全量重训。

这些是历史研究脚本，尚不是一个命令即可重建的训练管线。`train_submission.py` 会拒绝覆盖已存在的 `final.pt`。已有本地提交候选位于 `artifacts/submission_candidate/model.onnx`；克隆仓库不会包含它。

## 文档

- [数据策略](docs/DATA_STRATEGY.md)
- [初版模型设计](docs/MODEL_DESIGN.md)
- [数据审计报告](analysis/REPORT.md)
- [独立实验用法](experiments/README.md)
- [比赛参考资料位置](docs/reference/README.md)

原始数据、权重、缓存和比赛 PDF 保留在本地并由 `.gitignore` 排除。项目尚未添加开源许可证。
