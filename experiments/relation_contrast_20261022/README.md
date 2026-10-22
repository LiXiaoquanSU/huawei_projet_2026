# 跨通道关系与短时对比

先读PLAN.md和../../docs/SPECTRAL_DIVERSITY_20261022.md（两阶段联合报告）。本阶段全部实验在D:/huawei2026-experiments/relation_contrast_20261022保存，最终提交模型来自第二阶段。

从项目根目录依次执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/relation_contrast_20261022/check_design.py
.\.venv\Scripts\python.exe -X utf8 experiments/relation_contrast_20261022/train.py
.\.venv\Scripts\python.exe -X utf8 experiments/relation_contrast_20261022/deliver.py
.\.venv\Scripts\python.exe -X utf8 experiments/relation_contrast_20261022/finalize.py
```

依赖global_local_20261021/network.py、structure_events_20261021/train.py和src。训练复用已有metrics，修改结构后不得复用同名实验目录。固定5轮和预设排序。check_design验证真实缓存输入、零输入、梯度和相关矩阵对恒定偏移的不变性；后者仅适用于相关系数，不适用于带均值/能量/原分支的完整模型。

本轮实际完成18次拟合后，因开发效果较差，转入spectral_diversity_20261022；本阶段未运行deliver/finalize，不将其列为优先提交。上述导出命令仅供独立复核，最终交付以第二阶段报告为准。逐例预测和训练/验证源记录清单保留，未使用隐藏标签。原模型均保留。
