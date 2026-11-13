# Huawei TechArena 2026：Wi-Fi CSI 人数识别

根据600帧CSI和到达时间间隔，预测房间内0—4人。

## 最终提交

**提交根目录 [model.onnx](model.onnx)**。它是已报告公开平台成绩 **0.54** 的原版G′，不是后续研究候选。最终文件于2026-11-13选定并逐字节核对；私有测试成绩未知，本仓库不声称达到0.60。

- 大小：83,681,786字节，权重内嵌，无外部数据文件。
- SHA-256：`2b189c950412d0d770d82a0f118ebd95699e18b5b6706f56a8539c59a04b4534`
- 身份记录：[final_submission.json](models/final_submission.json)。选择文件不表示已经在网站上传。

## 使用

推理检查只需安装CPU依赖：

```powershell
python -m pip install numpy onnx onnxruntime
python src/check_model.py model.onnx
```

可用`--sample path/to/sample.npz`提供包含`frames`和`intervals`的真实输入；没有样例时使用全零输入，仅核对接口。输入`frames`为float32 `[600,1,3,1024,2]`（实部、虚部），`intervals`为float32 `[600]`（微秒）；输出`logits`为5个分数，取argmax得到人数。单次本地检查不代表老师机器的完整耗时或准确率。

## 代码结构

|目录|用途|
|---|---|
|`src/`|预处理、原四路线模型、整理后的训练与导出、ONNX检查|
|`experiments/`|历史实验源码及轻量结果，保留依赖供复核|
|`tools/`|模型核验、成绩登记与分析工具|
|`manifests/`|训练窗口清单，不含原始CSI|
|`models/`|最终文件身份、平台成绩和模型注册表|
|`docs/`|方法说明、实验结论与历史记录|
|`handoff/run037_20261025/`|原run训练资料和源码快照；大权重不纳入Git|

训练环境见[requirements.txt](requirements.txt)。研究脚本常引用本机D盘缓存和旧实验检查点，需要相应数据才能运行；不声称所有历史实验开箱即复现。原始数据、检查点、实验缓存及队友交接包保留在本地，未上传。

## 方法与结果

最终模型保留原0.37权重：静态幅度、动态幅度、相位关系及时间特征四条路线；结合类别偏置、接收通道排列/时间反转，以及类别保护路由达到已报告0.54。重训原结构不保证复现同一权重，整理版训练脚本不是原run的逐字节快照。

- [最终模型的结构与融合规则](docs/TEAMMATE054_20261025.md)
- [原run资料与复现边界](handoff/run037_20261025/README.md)
- [项目经验总结](docs/SUMMARY.md)
- [最后一轮实验结论](docs/RESEARCH_STATUS_20261111.md)
- [队友实验包复核](docs/CLAUDE_HANDOFF_REVIEW_20261110.md)
- [平台成绩记录](models/platform_scores.csv)

`docs/`中的日期报告是历史快照，“待测”“当前首选”等措辞只对当时有效。当前提交选择以本页和`models/final_submission.json`为准。训练内准确率、开发验证和平台成绩严格区分。

## 最终整理检查

2026-11-13完成282个Python文件语法检查、首页链接检查和最终ONNX结构/推理检查。已删除140个可再生成缓存文件，历史首页与整理前差异留在本地归档。模型哈希未变。

本机单个既有样例两次计时约0.97/0.93秒，高于0.7秒门槛；这不是老师机器的复测，不能声明本次本机速度合格。保留已报告平台0.54的原文件，未为本地计时改写它。详见[检查记录](docs/FINAL_VALIDATION_20261113.json)。
