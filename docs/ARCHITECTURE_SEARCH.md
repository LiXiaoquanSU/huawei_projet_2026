# 本轮架构研究结果

> 2026-10-20 更新：频率集合注意力版平台成绩为 **0.25**（用户报告），当前最佳仍为原版 **0.37**。本页编号保留为文件标识，已撤回频率集合版的优先推荐。新实验已暂停；研究代码移至 `_archive/research_20261020/research/`。

已完成 41 个 A 开发训练配置，涉及 30 个命名变体（含相同结构的预训练对照）。A 达到 70% 的配置 0 个；A 与 B 均达到 70% 的配置 0 个。训练准确率不计入此门槛。

已保存 16 个模型文件（含对照），精选 8 个进入 SUBMIT 目录。频率集合注意力版已实测 0.25，撤回其优先推荐；其他新文件平台成绩未知；原平台 0.37 文件哈希核对未变。

## 提交文件

| 源文件夹（提交请用 SUBMIT 编号文件） | 对应结构 A / B | 整房间均值 | 全量轮数 | 大小 | 最慢样例 CPU 均值 |
|---|---|---|---|---|---|
| `01_frequency_set_original_e4` | 47.3% / 40.5% | 22.3% | 4 | 55.73 MB | 0.038 s |
| `85_roomcal_random_e13` | 39.1% / 43.6% | 27.4% | 13 | 4.09 MB | 0.027 s |
| `92_best037_frequencyset_half_storage` | — / — | — | original tested .37; new frequency_set 4 | 70.86 MB | 0.133 s |
| `86_dual_pretrained_linear_probe_e3` | 31.8% / 36.4% | 27.2% | 3 | 3.49 MB | 0.027 s |
| `03_lstm_cnn_original_e11` | 47.3% / 36.8% | 24.1% | 11 | 70.79 MB | 0.061 s |
| `04_direct_original_e4` | 47.3% / 36.8% | 23.9% | 4 | 78.22 MB | 0.079 s |
| `95_tested037_teammate031_blend` | — / — | — | original tested binaries | 92.04 MB | 0.049 s |
| `70_baseline_early5` | 37.7% / — | — | 5 | 78.63 MB | 0.063 s |

包含复核、整房间、独立支持协议与全量重训，共记录 108 次有监督／微调训练运行；另外完成 40 轮无标签重建预训练。未将同一记录的窗口随机拆到训练和验证两侧。

## 本轮结论

- 记录留出最高 A 为 48.6%，最高 B 为 43.6%，没有达到 70%–90% 目标。A、B、整房间检查的领先者不同，没有稳定统一胜出的新模型。
- 本轮 LSTM、GRU、HMM、TCN、状态空间及注意力替换均未形成稳定优于基准的证据；这只涉及本次数据和配方。
- 频率集合版 A/B 为 47.3%/40.5%，适合作为原四路线静态 CNN 的结构对照；环境参照随机版 A/B 为 39.1%/43.6%，但新增无标签支持校准没有稳定收益，因此仍是探索候选。
- 直接微调的双模态重建预训练版 B 为 18.2%；冻结编码器后 B 为 36.4%，严格整房间五类平均召回约 33.2%，说明微调方式需要单独检查，重建损失下降不能直接当成人数识别成功。
- 训练诊断可超过 90%，记录和环境留出的结果仍低，当前证据更支持泛化受限。最终需要平台比较；保存的原 0.37 文件是已知成绩对照。

候选位于 `models/architecture_batch_20261019/`，统一提交入口为其中 `SUBMIT/`。A/B 成绩属于从头训练且留出记录的对应结构；全量重训使用了全部标签，无法再计算独立准确率。早期第 5 轮文件只引用该轮 A 历史分数，B 留空；原已评测二进制组合没有可信本地留出分数。每个文件的分数适用范围见 validation_context.json。

## 开发对照

| 结构 | 训练方案 | A 选中轮数 | 训练诊断 | A 开发留出 | B 固定预算复核 |
|---|---|---|---|---|---|
| baseline | original | 11 | 96.6% | 48.6% | 37.3% |
| bigru | original | 6 | 88.4% | 47.7% | 29.1% |
| baseline | cdf | 10 | 93.4% | 47.3% | — |
| direct | original | 4 | 80.1% | 47.3% | 36.8% |
| frequency_set | original | 4 | 74.6% | 47.3% | 40.5% |
| lstm_cnn | original | 11 | 92.5% | 47.3% | 36.8% |
| bigru | supcon | 5 | 79.3% | 46.8% | — |
| hmm | original | 6 | 90.1% | 46.8% | — |
| inception | original | 10 | 92.9% | 46.4% | — |
| tcn | original | 10 | 96.6% | 45.5% | — |
| time_ssm | original | 6 | 91.5% | 45.5% | — |
| dc_lstm | original | 1 | 36.9% | 44.5% | 18.2% |
| hurdle | original | 10 | 92.1% | 44.1% | — |
| baseline | supcon | 7 | 86.8% | 43.6% | — |
| bigru | cdf | 7 | 92.2% | 42.7% | — |
| dual_random | original | 10 | 70.6% | 42.7% | 35.0% |
| statistics | original | 8 | 88.8% | 42.3% | — |
| attention_fusion | original | 17 | 96.5% | 41.8% | — |
| baseline | natural | 6 | 92.8% | 41.8% | — |
| bigru | natural | 2 | 63.1% | 41.8% | — |
| spectrum | original | 7 | 87.9% | 41.8% | — |
| covariance | original | 3 | 84.7% | 41.4% | — |
| lstm | original | 4 | 82.4% | 41.4% | — |
| no_style | original | 9 | 89.3% | 41.4% | — |
| dc_lstm_absolute | original | 6 | 48.1% | 40.5% | — |
| hybrid_phase | original | 10 | 79.6% | 40.5% | — |
| patch_transformer | original | 5 | 51.8% | 40.5% | — |
| absolute | original | 1 | 36.6% | 40.0% | — |
| conv_attention | original | 4 | 80.1% | 39.5% | — |
| roomcal_pretrained | original | 1 | 43.4% | 39.1% | 18.2% |
| roomcal_random | original | 13 | 85.3% | 39.1% | 43.6% |
| patch_mixer | original | 2 | 39.0% | 38.2% | — |
| direct | ce | 12 | 95.4% | 37.7% | — |
| roomcal_pretrained | low_lr | 5 | 38.7% | 37.7% | 37.3% |
| static_phase | original | 16 | 87.4% | 37.3% | — |
| cnn_gru | original | 1 | 35.4% | 36.4% | — |
| dual_pretrained | original | 1 | 38.5% | 36.4% | 18.2% |
| dual_pretrained | low_lr | 3 | 47.2% | 34.5% | 27.3% |
| roomcal_pretrained | linear_probe | 16 | 44.6% | 32.7% | 40.0% |
| dual_pretrained | linear_probe | 3 | 42.8% | 31.8% | 36.4% |
| gaussian_hmm | original | 1 | 40.7% | 30.9% | — |

## 整房间留出

| 结构／方案 | a1 | c1 | g1 |
|---|---|---|---|
| baseline/original | 13.8% | 20.8% | 28.8% |
| dc_lstm/original | 33.3% | 16.7% | 58.8% |
| direct/original | 25.8% | 21.3% | 24.7% |
| dual_pretrained/linear_probe | 15.4% | 25.0% | 41.2% |
| dual_pretrained/original | 8.3% | 15.4% | 0.0% |
| dual_random/original | 18.8% | 0.0% | 0.6% |
| frequency_set/original | 16.2% | 23.7% | 26.8% |
| lstm_cnn/original | 21.3% | 25.8% | 25.3% |
| roomcal_pretrained/low_lr | 16.7% | 0.4% | 32.1% |
| roomcal_pretrained/original | 14.6% | 16.2% | 0.0% |
| roomcal_random/original | 21.7% | 6.7% | 53.8% |

## 如何理解

A 与 B 都以整条采集记录隔离；同一记录的窗口不会横跨该模型训练与验证。A 用于选结构与轮数，B 按 A 选定轮数从头训练，只评价末轮。A/B 是反复使用过的旧数据开发交叉验证，不是全新盲测；比较越多，开发高分越可能乐观。整房间留出更难，而且人数和房间在数据中相关，不能直接等同平台表现。

各类召回、混淆矩阵、逐窗口预测、训练与验证记录清单、配置及源文件哈希都保存在 `runs/architecture_search/development/`。接口预检不能代替准确率；最终候选另有隔离 CPU 检查、嵌入权重检查、数值误差及文件哈希。

论文依据和每项设计的动机见 `_archive/research_20261020/research/PAPERS.md`，评价约定见 `_archive/research_20261020/research/PLAN.md`。没有达到目标的结果也保留，不将其改写为成功。

## 无标签重建预训练

使用 145 条官方无标签记录的 1,160 个预定窗口，固定 40 轮，重建损失从 0.8158 到 0.2554。重建损失不是人数准确率；微调效果由上面的随机／预训练对照判断。未使用有标签验证记录做预训练。

## 独立无标签校准记录的补充验证

每个留出房间按记录 ID 字典序取前三分之一作为无标签校准支持，其余记录作为测试；支持记录和测试记录不重合，支持人数标签不参与训练或参照估计。下表同时把严格房间留出模型限制到相同测试记录，避免不同测试子集直接比较。此项模拟允许目标环境无标签数据的场景，不是严格无目标环境输入的房间泛化。

| 模型 | 房间 | 测试窗口 | 相同测试子集、无该房间参照 | 加入独立校准记录 |
|---|---|---|---|---|
| roomcal_pretrained/low_lr | a1 | 160 | 12.5% | 12.5% |
| roomcal_pretrained/low_lr | c1 | 160 | 0.6% | 0.0% |
| roomcal_pretrained/low_lr | g1 | 240 | 37.1% | 17.1% |
| roomcal_pretrained/original | a1 | 160 | 13.1% | 12.5% |
| roomcal_pretrained/original | c1 | 160 | 12.5% | 12.5% |
| roomcal_pretrained/original | g1 | 240 | 0.0% | 0.0% |
| roomcal_random/original | a1 | 160 | 31.2% | 21.9% |
| roomcal_random/original | c1 | 160 | 10.0% | 13.1% |
| roomcal_random/original | g1 | 240 | 45.8% | 24.2% |

数据对齐复查：960 个窗口的标签与清单一致；48 条记录各抽取一个固定窗口，原始输入与缓存的记录对应检查通过。完整结果为 `runs/architecture_search/alignment_audit.json`。

本地官方题目明确：测试房间不出现在有标签训练集中；允许无标签集与测试环境存在房间重合。因此整房间留出是重要参考，A/B 同房间新记录分数不能代替它。

运行中修复了一次 CSV 日志列不一致问题：固定预算的最后一轮新增验证列时旧写入器报错。基准 B 第 11 轮检查点已先保存，后来只恢复评价，没有重新选点或增加训练；训练损失和窗口读取函数未改，记录见 `runs/architecture_search/logging_repair/`。
