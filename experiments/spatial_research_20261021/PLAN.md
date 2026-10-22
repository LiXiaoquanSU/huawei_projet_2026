# 空间表征研究：0.38反馈后

用户反馈固定序数读出0.38，低于两个原0.39，已按SHA登记。本轮保留原分类/序数平均，不再据本地读出高分宣称平台提升。60%是跨房间开发目标，不是可保证成果，不以训练准确率充数。

## 文献核对

1. Cha等《Meta-Learning-Based People Counting and Localization Models Employing CSI from Commodity WiFi NICs》(2025)，https://arxiv.org/html/2502.03117v1 ：幅度及校正后实虚部输入CNN、FC分类，MAML面向新环境适配。其目标环境适配条件与本题无目标标签不同；可借鉴预处理/环境分组，不能搬用效果。
2. Custance等《Parameter-Efficient Domain Adaption for CSI Crowd-Counting via Self-Supervised Learning with Adapter Modules》(2026预印本)，https://arxiv.org/html/2601.02203v1 ：1D ResNet、SE、瓶颈adapter、对比预训练及少样本适配，进/出/无事件经有状态机器累计人数。本文任务与本题固定三秒绝对人数不同；不把WiAR活动准确率当人数准确率，不移植进出状态机。
3. Dai等《Attentional Feature Fusion》(WACV2021)，https://arxiv.org/html/2009.14082v2 ：局部逐位置通道注意力与全局通道上下文共同生成融合权重，适用于残差/跨层融合。改编为原CNN残差与shortcut的自适应选择，保留GN、SiLU，初始gate=.5并乘2以匹配原残差和；不是原论文复现。
4. Gao等《Res2Net》(TPAMI)，https://arxiv.org/abs/1904.01169 ：块内通道分组、逐级残差连接形成多尺度感受野。本轮在原CNN下采样后分4组逐级3x3卷积，再1x1融合；不是把CSI当照片解释，也不引用图像分类分数作预期。
5. 《Crowd counting with WiFi sensing based on iterative attentional feature fusion》(2025)，https://www.sciencedirect.com/science/article/abs/pii/S0140366425002026 ：检索摘要涉及静态/动态混合人数及幅相融合；全文访问403，不能声称核对了完整网络/划分。具体实现依据可访问的AFF原论文。

## 三个独立变体

aff_context：三路空间CNN残差块用AFF选择残差/shortcut；时间路线仍为已获0.39的context_events。res2_context：同三路改块内分组多尺度。pyramid_context：保留原卷积，以第二/三层的4个有序频率区域均值、标准差、最大值形成256维浅层证据，有界门控补充末层向量，检验细粒度结构是否被末层汇总压掉。

只用现有官方缓存；不虚构人体位置/速度，不复制天线，不改变时间采样。初始化/训练/loss/采样与原五轮协议一致，最终输出仍原等权概率。研究不按100MB筛掉结构，导出时再满足约束。

## 预设流程

原context/cross两个种子三折结果复用，至少重算context seed2026 a1 checkpoint确认混淆矩阵；不算新拟合。三变体seed42各a1/c1/g1整房间留出9次，按min(房间均值,宏召回)、最差房间、NLL选前2，各seed2026三折+seed42 e1，共8次；最终选两种子同指标更高者全48记录训练一次，共18次。即使未超过原对照也如实标研究候选，平台最佳仍0.39。不得按每折最佳epoch选取，固定5轮。

数据960窗口来自48记录；unknown只用于最终全量，开发完全排除留出房间，U/外部数据本轮不使用。开发反复使用、类别和房间混杂，必须报告逐类、逐房间与记录隔离，不能把局部60%冒充总体60%。当所有方案不接近60%时明确报告未达到，并给证据最好的下一提交候选，不无限调参声称必达。

输出D:/huawei2026-experiments/spatial_research_20261021；原0.37/0.38/0.39不覆盖，ONNX18、600帧接口、<100000000字节/<.7s/<6GB需实测。

数据复核：缓存48条原始记录中布局到物理房间映射为a→a1、b→b1、c→c1、e→e1、g→g1，unknown另列。因此当前留房间同时留布局，属于更严格的源布局迁移诊断；官方测试布局来自S/U布局族，不等同本地每折源S布局覆盖。保留现协议，不为达到60%更换成随机窗口或同房间测试；也不据该差异否定已经观察到的失败。
