# 第一轮监督基线训练

后续迭代结果见 `ITERATION_REPORT.md`；本文件保留首轮低分基线记录。

2026-10-18，使用RTX 4060 Laptop GPU，PyTorch 2.9.1+cu126。

## 设置

- 固定主折holdout_a1：33条已知房间训练记录、660个窗口；a1的12条记录、240个验证窗口。
- 排除unknown，不引入外部或官方无标签数据，不对验证集增强或重采样。
- 固定预处理在GPU生成特征缓存；预处理无可学习参数，也不拟合数据集统计。可学习网络在GPU训练，提交ONNX仍包含完整原始输入预处理。
- 种子42，batch32，AdamW，初始学习率0.001，weight decay 0.001。
- 交叉熵使用训练窗口频率的逆平方根类别权重、label smoothing 0.05；验证损失为未加权、未平滑交叉熵，二者不是同一种损失标度。
- 最多60轮，最少20轮，连续12轮未改善则早停。按验证准确率优先、验证交叉熵打破平局选检查点。

## 结果

- 实际训练24轮，最佳检查点在第12轮。
- 最佳检查点训练集评估准确率100%。
- a1验证GPU、CPU ONNX准确率均为16.25%（39/240）。
- 总预测为训练多数类1人的对照准确率16.67%（40/240）；均匀随机猜测的期望准确率20%。
- 0/1/2/3/4人的召回率分别为0%、45%、31.67%、3.33%、0%。4人窗口中44/60被预测为0人。
- 这说明当前结构/特征/训练策略组合在a1上泛化较差，与过拟合及域偏移一致；尚不能由单次实验确定是哪一支路造成。
- 验证集用于选模型，不是独立测试；240个窗口来自12条记录，不能视作240次独立采集。尚未训练辅助折，也没有平台成绩。

## 导出检查

- 最佳权重导出为单文件model.onnx，1,149,796字节，官方输入输出格式通过。
- 完整240个验证窗口上，CPU ONNX与缓存GPU预测类别100%一致。
- 与原始输入PyTorch CPU的最大绝对logit差为0.00356013；差异非零，不声称浮点逐位一致。本轮导出门槛为最大差小于0.01且GPU/ONNX全部验证预测一致，后续每个模型重新检查。
- 初始较紧的逐元素容差检查在最大差约0.00356处失败；随后保留同一检查点，完成全量误差和分类一致性核验，未重新训练或修改结构。
- 本机4线程CPU平均推理0.01011秒/窗口，P95为0.01166秒；不含数据文件解压时间，服务器耗时需以平台为准。

## 文件与复现

输出目录：`artifacts/baseline_a1_seed42/`。

- `best.pt` / `last.pt`：最佳与最后检查点。
- `config.json`：配置、GPU信息、模型源码及划分清单哈希。
- `roomsense_model_snapshot.py`：本次结构源码快照。
- `history.csv`、`training_curves.png`：训练过程。
- `metrics.json`、`confusion_matrix.png`：GPU评估及CPU混淆矩阵。
- `val_predictions_gpu.csv`、`val_predictions_onnx.csv`：逐窗口预测。
- `model.onnx`、`onnx_validation.json`：导出模型及CPU核验。

新跑一次使用新目录，防止覆盖本轮基线：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -X utf8 train_baseline.py --output artifacts\baseline_a1_seed42_repeat
```

仅从已保存检查点重新导出和生成报告：

```powershell
.\.venv\Scripts\python.exe -X utf8 train_baseline.py --finalize
```

后续优先做统计支路/ATI/动态特征消融，并在其他留出房间上验证；不把增加训练轮数作为首要改进。这轮低分模型保留作对照，不推荐作为正式竞赛方案。
