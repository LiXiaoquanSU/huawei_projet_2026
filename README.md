# CSI 人数识别

当前最高平台成绩为 **0.37**；SAM 已成功评测，成绩 **0.32**，不替换基准。训练和结构扩展继续暂停。

| 位置 | 用途 |
|---|---|
| `models/best_037/model.onnx` | 平台已评测 0.37，当前基准 |
| `models/sam_032/model.onnx` | SAM 版，平台已评测 0.32 |
| `models/*/final.pt` | 对应训练权重，用于恢复和重新导出 |
| `src/` | 6 个必要代码文件：模型、特征、工具、训练、导出、检查 |
| `manifests/train_windows.csv` | 48 条记录、960 个训练窗口的清单 |
| `docs/SUMMARY.md` | 模型介绍、训练方法和实验结论 |
| `docs/SAM_DOWNLOAD.md` | 本次平台报错及本地检查结果 |
| `docs/IMPROVEMENT.md` | 后续改进优先级及验证方法 |
| `_archive/history_20261019/` | 旧代码、实验、缓存、长篇报告的可恢复归档 |

提交时只选所需版本的 **model.onnx**。两份文件均约 82.06 MB，权重全部嵌入；本次整理没有改变它们的内容或哈希。

从本目录检查文件：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/check_model.py models/sam_032/model.onnx
```

保留的训练入口为 `src/train.py --method baseline` 或 `--method sam`，需要指定新的 `--output` 目录。数据默认在 `D:\huawei2026-data`，也可用 `--data-root` 指定。训练已暂停，本次未运行训练。

恢复权重后导出到新文件：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/export_model.py models/best_037/final.pt --output outputs/reexport/model.onnx
```

模型数学计算与旧代码一致；整理前后两份权重的原始输入及特征输入输出误差均为 0。原有 `.venv` 和 `.git` 保留。重新训练不保证逐字节复现原文件；已评测版本以保存的 ONNX 及其哈希为准。

Git 仓库只保存精简代码、说明、窗口清单与少量检查记录。ONNX、权重、原始数据、测试样例、环境、归档和比赛 PDF 留在本地，不随代码推送。没有本地测试样例时，检查／导出脚本使用全零占位输入，只用于检查接口，不能评估准确率。
