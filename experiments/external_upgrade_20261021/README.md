# 外部数据升级训练

训练前计划在PLAN.md。全部重型数据和训练结果在D:/huawei2026-experiments/external_upgrade_20261021，现有平台0.37文件不覆盖。

使用项目.venv Python，顺序：

1. `prepare.py`：RadioPoints/SHARP真实链路时间序列+Brescia原v2训练缓存，保存external_sequences.pt。
2. `run.py --stage screen`：四变体、seed42三折，12次训练。
3. `finish.py`：选定外部变体与source_control的seed2026三折复验，固定融合对照、全量训练与独立CPU导出检查。已完成拟合按metrics.json恢复，不覆盖重跑。
4. `verify.py`：原始输入/缓存一致性、链路排列不变性、源/验证隔离、全48记录覆盖、旧模型哈希检查。
5. `report.py`：报告与新研究候选登记。平台成绩保持空值，不改已确认最佳。

模型在net.py，不修改src。共享时间路线只接真实数据，SHARP活动仅用于辅助监督，没有给官方样本伪造静止/慢动/快动标签。新增变量天线数的集合汇总经过排列不变性检查；输出仍为老师要求的5类。

最终试提交副本固定为submissions/external_upgrade_20261021/model.onnx，文件身份与结果以该目录metadata.json、DATA_SOURCES.md及docs/EXTERNAL_UPGRADE_20261021.md为准。根目录model.onnx是旧文件，不是本轮输出。只有本机资源合格不代表隐藏测试准确率提高。
