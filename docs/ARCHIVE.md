# 归档与恢复

归档目录：`D:/huawei2026-experiments/project_archive_20261023_044`。

manifest.json 保存每个原路径、SHA256、大小和blob位置。blobs/按内容哈希去重存储，文件名本身不再表示原模型名。不要直接凭blob文件名挑模型；先查清单。

本次备份3186个文件，逻辑内容12.88GB，去重后9.38GB；逐个新blob复制后核对哈希。移除工作目录3147个旧文件，按文件大小约12.30GB，实际磁盘可用空间增加约11.72GB。硬链接和重复文件会使两个数值不同。

## 恢复示例

先预览旧整体训练代码：

```powershell
.\.venv\Scripts\python.exe tools/restore_archive.py --path experiments/joint_retrain_20261023
```

确认路径后执行同一命令并加 `--apply`。恢复器只向当前项目内写文件，验证归档哈希，遇到已有不同内容时拒绝覆盖。恢复旧失败代码不会自动启动训练。

已归档模型可按成绩表 original_path 查找，例如：

```powershell
.\.venv\Scripts\python.exe tools/restore_archive.py --path models/joint_retrain_20261023
```

归档包括整理前的README、完整registry、所有旧报告、实验代码、旧模型、提交副本、历史C盘归档和工作树差异。`.git`、虚拟环境和原始data未复制进去；这些内容未被本轮删除。D盘既有原数据和实验训练记录保持原位。

历史registry原文件路径为 models/registry.json；如需读取旧版本，直接按manifest定位blob，复制到另外的文件名查看。不要用旧registry覆盖当前登记。

当前未评测候选均已归档，暂停等待队友升级；不保留活动提交队列。
