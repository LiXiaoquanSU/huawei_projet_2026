# 训练差异实验

平台基准为已测 0.40 的原 cross42/context2026 固定概率融合。

本轮固定 context_events 网络、原960个非重叠窗口、5轮训练、原采样和优化器，对比：

- physical：原复合损失与原直接/序数融合读出。
- direct：直接分类 CE + 0.15 专家 CE；只用直接分类读出。

配方是“损失与对应推理输出”的整体对照，不是孤立某一损失系数的消融。两臂均做 seed314/2026 与三房间开发折，按新搭档与原 cross42 固定等权融合评估。部署种子预先固定314。开发后单独报告e1压力测试，不用它挑配方。训练验证源记录互斥，unknown只可进入最终全数据训练。共15次拟合。

运行顺序：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/training_diversity_20261023/train.py
.\.venv\Scripts\python.exe -X utf8 experiments/training_diversity_20261023/deliver.py
```

依赖本机 D 盘原特征缓存、历史验证预测、原模型与检查输入。输出在 `D:/huawei2026-experiments/training_diversity_20261023`。不覆盖原0.40。仅得到新平台成绩后才能确认收益。
