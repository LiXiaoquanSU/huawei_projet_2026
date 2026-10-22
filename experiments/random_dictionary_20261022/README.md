# 固定随机字典独立对照

ROCKET启发的稀疏多通道随机卷积与加权岭分类器。见PLAN.md。没有旧0.39骨干、没有梯度训练特征提取器，训练折标准化与lambda=10预设。

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/random_dictionary_20261022/train.py
```

实际8次诊断完成，两种子三房间均值约9.98%，不全量训练、不导出。`--full`仅供显式后续复核，本轮没有使用。完整联合报告位于../../docs/FRESH_BACKBONES_20261022.md。权重、源记录清单、逐例预测在D:/huawei2026-experiments/random_dictionary_20261022。

预计算的features仅来自固定随机变换，没有拟合数据；标准化和岭系数每折只看训练记录。更换核定义必须使用新的实验目录，不能重用features_s*.pt。
