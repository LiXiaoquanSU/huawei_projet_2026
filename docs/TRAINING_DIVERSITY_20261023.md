# 训练差异与互补模型实验

固定 context_events 成功结构，对比 physical 原复合损失与 direct 直接分类。direct = 五分类 CE + 0.15 专家 CE，去掉序数、JS 与信号辅助损失，同时推理只用直接分类概率，避免未训练的序数分支污染输出。其余数据、采样、5轮训练、优化器和每轮更新数保持原配方。

每臂 seed314、2026 × a1/c1/g1，12 次拟合；选定配方后固定 seed314 部署，两个配方另做 e1 压力测试，最后选中配方全48记录训练，共15次。训练/验证按源记录和房间隔离；unknown 不进开发。没有使用外部数据或密集滑窗。没有挑最高验证epoch。

比较目标是新 context 与原 cross42 固定等权融合，与原 cross42/context2026 组成的0.40模型在历史折上比较。选择依据为两种子平均 min(房间均值,宏召回)，种子314预先固定。历史折重复使用，不能当新独立测试。

|配方|融合房间均值|融合宏召回|最差房间|
|---|---:|---:|---:|
|physical|29.47%|31.53%|22.92%|
|direct|29.11%|32.10%|23.75%|

原0.40固定组合在相同历史折均值29.49%。新候选选中 `physical`；详细逐房间纠错/破坏数量、混淆矩阵、e1压力测试见本地 JSON 和 D 盘 selection.json。两种子对比中原 cross42固定，所以并不是两个独立验证集。e1压力测试单模型准确率：physical 21.67%、direct 26.67%；该结果在选配方后产生，没有反向用于挑选，也不代表完整融合模型的e1表现。

新候选：`models/training_diversity_20261023/01_cross039_context_physical_seed314_equalprob.onnx`，78.24 MB。SHA256 `0d1ccfe880edf46981ab38f8e033a535f93d20d18ac2ed42679d615de1e4129b`。新模型平台未知，不赋予0.40成绩，不保证提高0.05。原最佳完全保留。

采用已测0.40相同的半精度权重存储/还原float32计算方案。三输入CPU资源和数值检查通过；另960窗新搭档舍入数值检查最大概率差0.0002516，类别变化0个；此处不是准确率评估。

复现顺序：`python experiments/training_diversity_20261023/train.py`，随后 `python experiments/training_diversity_20261023/deliver.py`。需要既有D盘原特征缓存、原始最佳模型、历史验证预测和smoke/gap检查输入。所有拟合记录在 `D:/huawei2026-experiments/training_diversity_20261023`，权重不重复上传GitHub。
