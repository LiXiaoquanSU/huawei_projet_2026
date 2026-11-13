# 跨天线同步是否是人数证据：2026-10-29

## 问题与方法

上轮天线图早期权重几乎不依赖端点绑定。本轮先验证这种对应关系是否在当前模型可访问的缓存特征中可检测，再验证它是否具有人数相关性，而不是先把同步判别强加给人数模型。

使用全部48条标注记录、960个非重叠窗口。幅度先对齐到32个频率组，与相位组对应。对三对天线，在三个时间尺度计算相关系数、绝对相关、频率间离散程度和能量轨迹相关，跨天线对对称汇总为48维。幅度—相位组使用幅度变化能量与标准化相位特征变化能量，不声称是原始物理相位或绝对人体运动。

正样本：同窗口同步。负样本1：裁剪相差29个缓存时间点的片段，不循环拼接；负样本2：使用同记录相隔7个非重叠窗口的另一信号。每种二分类正负数相等，机会水平50%。每条记录所有正负样本都留在同一个训练/校准/测试分区，保持房间和人数标签相同。

控制：从每根天线、每个时刻减去它自己的跨频率均值，去掉宽频共同变化；不在三根天线之间相互减均值，避免人为制造接收通道负相关。该操作也可能去掉人体宽频变化，不能当成纯设备噪声隔离。相位特征与人数摘要未改。

三折整记录、四折留房间，标准化只拟合训练部分。逻辑回归C∈{0.01,0.1,1}仅由源校准均衡NLL选定。两组控制共294次轻量拟合。同步判别与人数判别是不同任务，不能把前者90%以上解释为人数识别成功。

## 同步判别

|控制|协议|证据|错配方式|同步判别正确率|空房判别正确率|按整记录重采样95%区间|
|---|---|---|---|---:|---:|---|
|original|record|rx|shift|90.16%|92.27%|[0.8693, 0.9307]|
|original|record|rx|other_window|89.48%|90.91%|[0.8615, 0.925]|
|original|record|amp_phase|shift|89.90%|91.59%|[0.8745, 0.9219]|
|original|record|amp_phase|other_window|89.84%|90.23%|[0.876, 0.9198]|
|original|room|rx|shift|87.10%|89.75%|[0.8449, 0.8972]|
|original|room|rx|other_window|85.57%|88.25%|[0.8341, 0.8767]|
|original|room|amp_phase|shift|87.22%|88.50%|[0.8477, 0.8966]|
|original|room|amp_phase|other_window|87.22%|87.50%|[0.8477, 0.8983]|
|frequency_centered|record|rx|shift|86.04%|89.77%|[0.8245, 0.8922]|
|frequency_centered|record|rx|other_window|83.70%|87.27%|[0.8042, 0.8682]|
|frequency_centered|record|amp_phase|shift|96.35%|96.82%|[0.9526, 0.9734]|
|frequency_centered|record|amp_phase|other_window|95.52%|96.82%|[0.9453, 0.9646]|
|frequency_centered|room|rx|shift|82.05%|85.50%|[0.7943, 0.8466]|
|frequency_centered|room|rx|other_window|81.82%|84.75%|[0.7943, 0.8415]|
|frequency_centered|room|amp_phase|shift|95.51%|95.75%|[0.9437, 0.9659]|
|frequency_centered|room|amp_phase|other_window|94.38%|95.75%|[0.9307, 0.9557]|

record为整记录隔离，room为整房间隔离。空房的同步也可以来自共享硬件、环境或预处理产生的相关性；本实验没有标定信息来分解这些来源。

## 同步特征用于人数分类

|控制|协议|输入|正确数/窗口数|正确率|
|---|---|---|---:|---:|
|original|record|summary|279/960|29.06%|
|original|record|relation|366/960|38.12%|
|original|record|combined|357/960|37.19%|
|original|room|summary|200/880|22.73%|
|original|room|relation|112/880|12.73%|
|original|room|combined|162/880|18.41%|
|frequency_centered|record|summary|279/960|29.06%|
|frequency_centered|record|relation|381/960|39.69%|
|frequency_centered|record|combined|366/960|38.12%|
|frequency_centered|room|summary|200/880|22.73%|
|frequency_centered|room|relation|229/880|26.02%|
|frequency_centered|room|combined|189/880|21.48%|

summary为13维接收通道对称强度/时间摘要；relation为96维真实同步特征；combined为二者拼接。人数探针只输入真实同步样本，未将错配样本伪装成新的标注人数训练数据。

|相对摘要基线|协议|特征|准确率差|按整记录重采样95%区间|
|---|---|---|---:|---|
|original|record|relation|+9.06%|[-0.0188, 0.1948]|
|original|record|combined|+8.12%|[-0.0083, 0.1657]|
|original|room|relation|-10.00%|[-0.2398, 0.0262]|
|original|room|combined|-4.32%|[-0.1864, 0.0864]|
|frequency_centered|record|relation|+10.62%|[-0.0, 0.2063]|
|frequency_centered|record|combined|+9.06%|[0.001, 0.1771]|
|frequency_centered|room|relation|+3.30%|[-0.1353, 0.1818]|
|frequency_centered|room|combined|-1.25%|[-0.1511, 0.1148]|

区间按各折内完整记录重采样3000次，不能覆盖反复架构探索的选择偏差、全部随机种子或未知测试分布。没有新增盲测数据。

## 结论与后续选择

1. 当前预处理后的同步关系可被可靠检测，并非“数据完全没有对应信息”。

2. 空房中同步判别同样很高，因此同步不是人体特有证据。去掉跨频率共同变化也没有消除这一现象；不能简单归因于宽频增益，更不能宣称已分离人体与硬件。

3. 人数探针表现远低于同步判别，且跨房间收益不稳定。同步可能适合检查信号一致性，但本轮没有证明“训练同步辅助任务”能改进人数识别。也不能据轻量线性探针排除更复杂网络利用这些特征的可能。

4. 下一步应研究同一房间内部哪些变化随人数具有可重复关系。a1与c1共同覆盖1—4人，但c1没有空房，不能虚构统一空房基线；可先比较两房间1→2→3→4的变化方向，空房/有人另用具有0人数据的记录分析。带已知人数的房间参考只能用于机制诊断，不能偷偷给提交模型提供测试房间基线。

本轮没有新ONNX或平台提交。原模型完整保留。原始窗口、缓存或数据划分没有修改。代码：`experiments/synchrony_20261029/`；结果及逐样本概率：`D:/huawei2026-experiments/synchrony_20261029/`，其中`spectral_control/`保存共同变化控制。
