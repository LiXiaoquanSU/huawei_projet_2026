# 独立骨干实验复现

不是0.39加分支。确定性预处理相同，三个骨干、输入组织和单分类头从零设计训练。先读PLAN.md、../../docs/FRESH_BACKBONES_20261022.md。

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/fresh_backbones_20261022/check_design.py
.\.venv\Scripts\python.exe -X utf8 experiments/fresh_backbones_20261022/train.py
.\.venv\Scripts\python.exe -X utf8 experiments/random_dictionary_20261022/train.py
.\.venv\Scripts\python.exe -X utf8 experiments/fresh_backbones_20261022/deliver.py
.\.venv\Scripts\python.exe -X utf8 experiments/fresh_backbones_20261022/finalize.py
```

24次神经网络拟合+8次独立随机字典/岭分类，共32次。原context_control也12轮但保留原复合loss，不能当纯结构单因素实验。原5轮结果另列。所有全量模型训练分数不是测试分数。

D:/huawei2026-experiments/fresh_backbones_20261022保存权重/清单/逐例预测；random_dictionary_20261022为失败的第四路线，未导出。修改结构后不得复用同名已完成metrics。最终交付仍需ONNX独立CPU数值/资源检查；平台最佳0.39保持。
