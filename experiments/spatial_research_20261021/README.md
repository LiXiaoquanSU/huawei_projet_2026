# 空间结构研究复现

于2026-10-21启动、2026-10-22完成，目录保留启动日期。

先读PLAN.md与../../docs/SPATIAL_RESEARCH_20261021.md。训练固定三种结构，按房间隔离，两个种子复验，18次拟合。选择结果保存在D:/huawei2026-experiments/spatial_research_20261021/selection.json。

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/spatial_research_20261021/train.py
.\.venv\Scripts\python.exe -X utf8 experiments/spatial_research_20261021/deliver.py
.\.venv\Scripts\python.exe -X utf8 experiments/spatial_research_20261021/finalize.py
```

训练脚本复用已有metrics，不要修改网络后复用同目录。依赖experiments/global_local_20261021/network.py、experiments/structure_events_20261021/train.py及src；缓存位置见train.py引用的load实现。D盘需要写入权限。

最佳新候选pyramid_context seed42两种子跨房间均值27.57%，未超过原对照、未达60%。全量训练后的81.35%是训练数据重评，不是独立测试准确率。正式文件和提交副本哈希一致，平台成绩未知。CPU检查、源记录隔离审计见../../checks/spatial_research_20261021.json。

经验：Res2首种子29.57%在第二种子降至22.76%，不能凭单种子下结论；金字塔保留浅层频率细节也可能保留房间特征。三个新方案在当前五轮协议下均没有证据取代两份0.39。对高人数尤其4人的跨房间识别仍失败；优先研究可控的房间背景归一化及跨房间一致性，比较逐类召回，再考虑继续增加容量。此处是后续假设，尚未验证。
