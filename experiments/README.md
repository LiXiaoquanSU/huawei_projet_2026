# 独立实验入口

从仓库根目录运行，以便导入根目录模型模块并读取相对路径：

```powershell
python -m experiments.cache_phase
python -m experiments.physics_experiments
python -m experiments.projected_experiments
```

这些命令会生成缓存或运行实验，需先准备数据、GPU 环境和基线产物；不是安装验证命令。
`cache_phase` 依赖基线缓存；投影实验还依赖时频及相位缓存。输出仍保存在 `artifacts/`。

`phase_models.py`、`physics_model.py`、`projected_model.py` 等模型模块保留在根目录，供实验及公共训练代码导入。
