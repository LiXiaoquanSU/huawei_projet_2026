# 本地环境

已于 2026-10-18 验证。

- Python 3.13.1，项目虚拟环境：`.venv`。
- 官方依赖均已安装；精确版本见 `requirements-lock.txt`，`pip check` 通过。
- PyTorch 2.9.1+cu126（官方 CUDA 12.6 版）；RTX 4060 Laptop GPU（8 GB）GPU 训练验证通过。
- 当前 NVIDIA 驱动 546.92 已实际通过卷积前向/反向、Adam 参数更新测试；其他算子或运行时编译功能仍需随正式模型验证。
- VS Code 默认解释器已配置为 `.venv/Scripts/python.exe`。如果工作区此前手动选过其他解释器，请执行 `Python: Select Interpreter` 并选择此路径。
- 新建 VS Code 终端会启用 Python UTF-8，避免 PyTorch 导出日志触发 GBK 编码错误。

## 数据

位置：`D:\huawei2026-data`。

- `labeled/`：48 条记录。
- `unlabeled/`：150 条记录。
- 两份原始 ZIP 均保留，官方 SHA-256 校验通过。
- 全部索引对应的 CSI/ATI 文件存在；各数据集抽样数组形状、类型和有限值检查通过。

## 在 VS Code 的 PowerShell 终端运行

无需激活虚拟环境，也无需修改 PowerShell 执行策略：

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 example.py D:\huawei2026-data\labeled --output artifacts/smoke/smoke_model.onnx
```

真实数据测试已成功生成 7,393 字节的 ONNX 模型，预热后单次 CPU 推理约 0.007 秒。这是未训练的接口示例，耗时不代表后续训练模型的性能，也不代表评测服务器耗时。

## GPU 训练到 CPU 提交验证

```powershell
.\.venv\Scripts\python.exe -X utf8 verify_gpu_export.py D:\huawei2026-data\labeled
```

已通过 GPU 卷积前向/反向、真实样本上三步示例模型训练、CPU ONNX 导出、ONNX 检查和 CPUExecutionProvider 推理。GPU、PyTorch CPU 和 ONNX CPU 输出在设定数值容差内一致，20 次预热后 CPU 推理平均约 0.004 秒/样本。

`artifacts/smoke/gpu_smoke.onnx` 仅用于环境与接口测试，不是正式参赛模型。GPU 训练不影响 CPU 提交；正式模型仍须验证文件大小、输入输出、CPU 耗时和内存，最终成绩以平台评测为准。

第一版实际模型结构及内置预处理见 `MODEL_DESIGN.md`、`roomsense_model.py`。用 `python -X utf8 validate_model.py` 验证其GPU训练路径及CPU ONNX推理，结果保存在 `artifacts/model_design/`；这也只是未训练结构验证。

后续正式训练及改进结果见 `ITERATION_REPORT.md`。当前全量有标签提交候选为 `artifacts/submission_candidate/model.onnx`，相应CPU核验为同目录 `cpu_validation.json`。候选没有独立本地准确率；开发成绩属于 `artifacts/improved_fusion/holdout_*/` 中的独立留出模型。

如需重建 GPU 环境，先安装官方 GPU 版 PyTorch，再安装其余依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如需重新核验并解压原始压缩包（会重新写入同名数据文件）：

```powershell
.\.venv\Scripts\python.exe prepare_data.py D:\huawei2026-data
```
