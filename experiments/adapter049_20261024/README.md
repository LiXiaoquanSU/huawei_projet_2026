# 固定0.49校正流程的时间适配器

`run.py`：生成独立代理四视角编码缓存，运行最初的监督/一致性/同类对齐消融。`calibrated.py`：固定偏置，把监督目标对齐实际提交融合输出；加`--extended`比较同类对齐和困难组训练。`record_holdout.py`：整记录排除适配器拟合，明确旧主干已见标签。

`export.py CHECKPOINT --mix 1.0`：从已评分0.49图复用原编码结果，追加适配器和冻结读出，逐窗核查960个原始CSI/ATI输入；与独立PyTorch读出及原0.49输出对照。

对D盘导出目录运行`tools/check_candidate.py`，隔离CPU验证通过后，`publish.py`只复制一个候选到C盘并更新登记。所有大缓存、训练参数、其他候选留在D:/huawei2026-experiments/adapter049_20261024。不得覆盖已有平台成绩的候选。

固定选择：calibrated_consistent、75步、残差0.25、输出mix1.0。平台未知，不替代已评分0.49最佳。
