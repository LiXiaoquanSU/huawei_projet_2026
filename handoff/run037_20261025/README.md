# 原run核对与队友交接

主文件 final.pt 是原始0.37底座，不是新训练出来的替代品。其配置直接从checkpoint及原提交归档交叉确认。

- 随机种子42：Python/NumPy/Torch/CUDA及WeightedRandomSampler均设42；spectral/physical，5个epoch，来自a1/c1/g1开发最佳epoch[1,19,5]的中位数。全量训练重新随机初始化，不是把三个fold权重平均。
- 比赛有标签48条记录全部训练，960个600帧不重叠窗口，stride600；包含3条room=unknown但人数标签为1的记录。每条20窗。没有外部数据、官方U、重叠滑窗或人为动作/房间面积标签用于这个原run。
- data/records48_sha256.csv给出全部96个CSI/ATI文件名、标签、大小和本次重新核验SHA256；data/train_windows.csv给出960窗及确切原始顺序。原始CSI/ATI大文件不在本包。
- WeightedRandomSampler按类别→房间平衡、有放回每epoch抽960次；unknown按记录identifier作组，不把它断言为同一物理房间。因此一轮不保证每个窗口都出现一次。batch32，AdamW lr0.0004、weight_decay0.01，梯度裁剪3；全量refit固定5轮，没有scheduler或early stopping。
- 损失=direct NLL +0.15专家CE +0.5条件序数BCE +0.1双头JS +0.1六项信号相干代理MSE；原run没有label smoothing。

不能只回答“没有增广”：原训练forward内实际包含：
1. 幅度/相位高斯噪声std0.02；5%频率格mask，沿时间共享，128格幅度用32格mask每格重复4次。
2. 静态分支前两个残差块后的MixStyle：批内置换，Beta(0.2,0.2)，训练时启用。
3. 四路线独立25%随机丢弃；全丢时恢复全保留；另有共享读出Dropout0.4与时间FFN Dropout0.1。
原run没有SAM、MixUp、CutMix、训练时Rx置换/时间反转、U伪标签或外部数据微调。后来的其他实验不能混算入此run。

original_source/下refit_structured.py、train_structured.py、structured_model.py、export_structured.py、validate_model.py来自原提交source_snapshot，逐文件校验归档hash；其余依赖来自同时期历史归档，不能保证全为运行当刻快照。详见provenance.json。

注意当前项目src/train.py是后期整理版，不应充当原run逐字节快照。原脚本依赖artifacts/tf_cache、phase_cache、baseline_a1_seed42的原特征银行；包内没有这些大缓存。直接从raw重算可能有浮点/批量/环境差异，不能承诺只凭seed42重训逐字节一致。final.pt不含优化器或RNG状态，不能精确续接历史优化轨迹。

当前0.54版本的区别：
- 原Gprime054：原底座权重按队友0.44文件的FP16存储精度匹配，使用队友给定五维偏置（估计过程未完整提供），6种Rx排列×正反时间，既有保护/融合规则及末尾1人+0.35。
- 选择性054：还复用seed42、paired方法第120步的相位/时间后缀，与初始权重按25%/75%插值并按原存储精度舍入；原Gprime最终预测2+保持，0/1区域使用适配专家。扩展checkpoint另放extensions/paired_seed42_step120.pt，仅含后缀state/initial，并非完整final.pt。
- 最新偏置054：在选择性054最终输出追加[-0.05,0,+0.15,-0.15,-0.15]，用户本次反馈仍0.54，非新的神经网络训练。原0.37 final.pt单独不包括这些后处理和额外后缀。

扩展paired训练仍只用同一960个有标签窗，无U/外部数据。seed42，源采样generator42、同房间配对generator20042；模型eval以冻结Dropout/MixStyle，训练最后phase.project、temporal.blocks.2、temporal.project。240步中保存40/120/240；实际选择120。AdamW编码器lr5e-5、weight_decay0.01、clip1，源batch16并额外8对同房间0/1；0.02标签平滑、0.1教师KL、0.02表示漂移、0.3配对binaryCE、0.1softplus排序。输入4视角，不等于原run训练时有TTA增广。历史原生特征缓存与实际ONNX预处理存在已发现的数值差异，不保证此旧扩展训练完全对齐最新ONNX口径。

本包提供证据和权重，不声称已完整复现队友未提供的偏置估计历史。平台成绩是用户报告，不是本次重新调用老师测试机所得。
