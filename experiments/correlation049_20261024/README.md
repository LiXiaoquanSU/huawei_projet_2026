# 独立相关性统计辅助分支

`run.py`提取相关矩阵谱矩、自相关和能量分布，比较有无绝对统计及softmax/条件序数读出；`replicate.py`复验第二种子和整记录排除；`no_clock.py`移除采样间隔summary，再运行`replicate.py --no-clock`。

最终固定：114维无绝对幅度/ATI统计、softmax小读出、种子42/2026平均、50步快照。`controls.py`比较常量预测并核查统计描述的Rx/时间变换；`record_ensemble.py`核查最终双头整记录排除。

`export.py`保留原0.49图，新读出仅作用于原预测2/3/4的输入，0/1原分数精确保留；融合概率0.5原模型＋0.25读出42＋0.25读出2026。导出时核查960个原始窗口及独立PyTorch计算。

对D盘export目录运行`tools/check_candidate.py`隔离CPU验证，再运行`publish.py`。检查通过才复制一个候选到C盘。全部缓存、参数和其他对照保存在D:/huawei2026-experiments/correlation049_20261024。尚未取得隐藏成绩。
