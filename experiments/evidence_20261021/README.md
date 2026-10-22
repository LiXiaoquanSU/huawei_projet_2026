# 0.39之后的结构与输出端研究

完成28次新拟合：第一阶段21次（原对照3、三结构筛选9、两方案换种子6、e1两次、全量一次），类别查询阶段7次（两种子三折、e1一次）。一次被中断的4轮训练另保留，不计完成拟合。输出头诊断与固定融合仅评估已有折内权重，不计新训练。

先读 ../../docs/AFTER_039_RESULTS.md。下一研究提交是 ../../submissions/ordinal_readout_20261021/model.onnx，平台未知。其权重来自已有context_events seed2026全48记录训练，只改变最终概率读出。本地28.66%→31.77%，不保证平台超过0.39。

network.py：幅相对应token、实际时间距离attention、组合；query_network.py：人数条件查询。preflight.py验证信号轴身份及时间性质。train.py使用原固定五轮训练，train_query.py独立阶段。不要将full权重用于开发。

diagnose_heads.py比较固定direct/ordinal/原平均；evaluate_equal_blend.py固定等权融合，无系数搜索。deliver.py导出第一阶段；export_ordinal.py导出推荐读出候选。summarize.py汇总并核对记录隔离、文件哈希和标签。deliver_query.py仅在查询模型通过预设规则时可用，本轮未通过，因此没有查询全量模型可导出。

完整记录：D:/huawei2026-experiments/evidence_20261021、query_20261021、ordinal_readout_20261021。原模型均保留。训练程序会跳过已存在metrics的拟合；如要改变模型或训练设定，须使用新实验目录，不能修改代码后把旧metrics当新结果。未来平台反馈后不可重新执行历史导出/登记脚本覆盖新成绩。
