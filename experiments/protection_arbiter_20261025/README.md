# 条件保护仲裁器

最终交付分支是`bayes.py`（原始视角对数赔率证据），不是下面的内部专家中间方案。
报告：`docs/PROTECTION_ARBITER_20261025.md`。待测文件：
`models/prior_free_arbiter_20261025/01_prior_free_evidence_arbiter.onnx`。
留房间278→298/880，来源676→647/960，平台未知；所有失败保留，已评分最佳仍0.54。

完成下面的`crossfit.py`后，最终分支运行顺序：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/bayes.py
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/export_bayes.py
.\.venv\Scripts\python.exe -X utf8 tools/check_candidate.py D:/huawei2026-experiments/protection_arbiter_20261025/prior_free/export
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/export_bayes.py --publish
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/report_bayes.py
```

以下保留为同轮中间实验说明，不是最终推荐导出指令。

目标：判断原054保护/融合结果什么时候值得被其他视角纠正。不是直接取消保护，不重新训练提交用的原0.37底座，也不新增人数偏置。

数据与防泄漏边界：

- 外层a1/c1/g1/e1整房间留出；该房间不进入对应任何新底座、校准或仲裁器训练。
- 源数据沿用旧fit/calibration记录划分。源fit再按整记录分3折：每个新底座仅训练其中2折，独立calibration记录拟合偏置/温度，预测未参与训练/校准的另一折。
- 仲裁器的纠错标签来自这些OOF预测；不得直接拿提交底座的训练内错误训练仲裁器。
- 仲裁器比较20份意见：原054、完整融合、6份双时间Rx意见、12份单视角意见。使用共享小网络预测相对改判效用；训练中改坏正确结果的代价为2，救回错误收益为1。
- 比较线性、关系MLP、带候选人数编码MLP。源OOF再分折选择意见范围/阈值；源元验证涉及重复开发，并非新的盲测。完全隔离的外层房间才是主要开发检验。
- 以不改判为默认候选。原054网络和原人数偏置不变。

从仓库根目录使用项目Python，按顺序运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/crossfit.py
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/learn.py
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/evaluate.py
```

只有外层验证支持，才生成实际提交底座对应的OOF样本并导出：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/crossfit.py --actual
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/learn.py --actual
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/export.py
.\.venv\Scripts\python.exe -X utf8 tools/check_candidate.py D:/huawei2026-experiments/protection_arbiter_20261025/export
.\.venv\Scripts\python.exe -X utf8 experiments/protection_arbiter_20261025/export.py --publish
```

已有缓存会复用，改配方时应新建目录。原始记录、12/15份新底座、OOF预测、小仲裁器及校验输出保存D盘。方案未过门槛时不能把原模型复制后称新候选；代码不会自动提交老师平台或推送Git。
