# 运动证据与外部数据实验

本目录是新增实验代码，不替换 `src/` 或两份平台0.37文件。计划见 [PLAN.md](PLAN.md)，幅度比补充方案见 [RATIO_PLAN.md](RATIO_PLAN.md)，数据边界见 [EXTERNAL_DATA_NOTES.md](EXTERNAL_DATA_NOTES.md)。结果以D盘各JSON和最终报告为准；平台成绩均需另行反馈。

脚本职责：

- `acquire.py`：公开Zenodo元数据、HTTP Range选择性提取0—4人MAT／配置，ZIP CRC检查与断点跳过。
- `prepare_external.py`：复数／时间戳解析、精确内容锚点查重、名义与实测时间区分、日期隔离的单天线幅度特征。
- `net.py`：原宽度骨干上的静态、慢变化、快变化、三分支和时间感知频谱适配器。
- `run.py`：固定房间／种子／更新预算的结构筛选与复核，不加载全标签模型参与开发。
- `ratio.py`：从原始三天线输入提取稳定幅度比并作独立对照。
- `transfer.py`：单天线外部预训练，以及同更新预算的官方源数据预训练对照。
- `finish.py`：等待结构实验结束后，顺序执行迁移、幅度比及条件复核，避免GPU争用。
- `export_candidates.py` / `export_ratio.py`：导出并调用独立CPU检查器验证。
- `report.py`：收集已完成结果、核对旧模型哈希，更新本轮报告和模型登记；运行前要求全部实验完成标记。

额外依赖：`h5py`（本机已安装3.16.0）；其他依赖沿用项目环境。原始数据、缓存、权重和逐折结果均位于：

迁移训练使用 `external/single_rx_features_v2.pt`、`windows_v2.csv`和`cleaning_audit_v2.json`，幅度公式与官方静态路线对齐。初版同名无v2特征作为清洗过程记录保留，没有用于迁移训练。

```text
D:/huawei2026-experiments/motion_external_20261021/
  external/          原始元数据、配置、MAT、清洗审计和单天线特征
  fits/              官方整房间与全48条训练结果
  pretrain/          单天线预训练及计算预算对照
  exports/           ONNX与独立CPU资源检查
  source_snapshot/   启动结构实验时的代码
  final_source_snapshot/ 完成时的全套代码和哈希
```

本机路径在脚本中明确固定，迁移到队友电脑前应统一修改数据／实验目录并重新登记配置。不要把同一轮已完成结果目录当作新实验输出；脚本会复用已有metrics，改变方案必须建立新实验目录。

D盘源码快照用于审计，不能从快照目录直接执行：脚本按项目相对层级寻找`src/`。复现时先恢复到项目的`experiments/motion_external_20261021/`层级，再设置新的实验输出目录。

不把插值视为新增真实采样，不复制天线，不用外部验证替换官方房间验证，不把信号变化分量标为真实动作标签。外部数据权重与官方-only候选分开标记，发布／赛事外部数据条件未由此代码确认。
