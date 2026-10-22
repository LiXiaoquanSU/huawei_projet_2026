# 谱多样性结构复现

接续relation_contrast_20261022第一阶段18次拟合；本阶段15次，合计33次。两阶段假设、论文边界见各自PLAN.md，最终结果见../../docs/SPECTRAL_DIVERSITY_20261022.md。

从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/spectral_diversity_20261022/check_design.py
.\.venv\Scripts\python.exe -X utf8 experiments/spectral_diversity_20261022/train.py
.\.venv\Scripts\python.exe -X utf8 experiments/spectral_diversity_20261022/deliver.py
.\.venv\Scripts\python.exe -X utf8 experiments/spectral_diversity_20261022/finalize.py
```

依赖global_local_20261021/network.py、structure_events_20261021/train.py和src。训练复用已有metrics，不要修改结构后重用同目录。D盘保存权重与逐例预测；导出检查要求第一阶段结果已存在，确保不遗漏失败方案。check_design验证零输入、梯度、原主体初始化一致和谱矩的正交变换性质；数学性质不是房间泛化证明。

新候选不继承原模型0.39成绩。全量训练数据的重评准确率不是独立测试。未经新平台结果不得称已提升。
