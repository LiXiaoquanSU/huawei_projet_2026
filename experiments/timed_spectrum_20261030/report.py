import json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=Path('D:/huawei2026-experiments/timed_spectrum_20261030')
probe=json.loads((WORK/'probe_results.json').read_text())['results'];neural=json.loads((WORK/'neural/results.json').read_text())['results'];assert len(probe)==56 and len(neural)==28
lines=['# 实际采样时间频谱实验（2026-10-30）','','从48条原始CSI/ATI记录提取960个非重叠窗口；输入只使用单窗口。7个倍频频段0.5—64Hz，每段4个投影频率；梯形时间权重、去时间均值，幅度与相对相位谱能量分别归一化，按接收通道/天线对做对称均值与标准差。低频分辨率受约3秒窗口限制；频段能量不能解释为独立人数。','','三个等维输入：实际时间600帧、等间隔600帧、实际时间4帧平均。后者高频可能混叠，是压缩对照，不是有效高频恢复。另有13维摘要基线。时间反转与零输入检查通过。','','## 224次轻量拟合','','每折每配置4个正则强度，仅源校准均衡NLL选择；线性及512维随机傅里叶映射分类器。所有拟合包括缩放只用源训练记录。','','|协议|输入|分类器|正确/总数|','|---|---|---|---:|']
summary={'probe':{},'neural':{}}
for group,folds in [('record',('record0','record1','record2')),('room',('a1','c1','g1','e1'))]:
 for mode in ('summary','actual','uniform','coarse'):
  for family in ('linear','rbf'):
   rr=[r for r in probe if r['fold'] in folds and r['mode']==mode and r['family']==family];a=sum(r['correct'] for r in rr);b=sum(r['total'] for r in rr);summary['probe'][f'{group}_{mode}_{family}']=[a,b];lines.append(f'|{group}|{mode}|{family}|{a}/{b} ({a/b:.2%})|')
lines+=['','## 28组频谱卷积网络训练','','两个输入×两个种子×7折，每组30轮，共840轮。约7×32频谱图输入三层卷积，两个频率区域均值/标准差池化，接摘要和128维分类头。保留源校准选轮与固定末轮。不是原有0.54权重续训。','','|协议|输入|种子|源选正确/总数|固定30轮|选择轮数|','|---|---|---|---:|---:|---|']
for group,folds in [('record',('record0','record1','record2')),('room',('a1','c1','g1','e1'))]:
 for mode in ('actual','uniform'):
  for seed in (42,2026):
   rr=[r for r in neural if r['fold'] in folds and r['mode']==mode and r['seed']==seed];a=sum(r['test']['correct'] for r in rr);b=sum(r['test']['total'] for r in rr);last=sum(r['last_test']['correct'] for r in rr);epochs=[r['epoch'] for r in rr];summary['neural'][f'{group}_{mode}_{seed}']=dict(correct=a,total=b,last=last,epochs=epochs);lines.append(f'|{group}|{mode}|{seed}|{a}/{b} ({a/b:.2%})|{last}/{b}|{epochs}|')
lines+=['','## 决策边界','','这些反复使用的开发划分不是盲测。时间处理改善是否有效，应看actual与uniform匹配差异，不能只对比更弱摘要。若不能跨种子和房间支持该假设，则不追加扩大此结构。未导出新提交模型；平台最佳仍0.54。所有特征、权重和逐窗口预测保存于`D:/huawei2026-experiments/timed_spectrum_20261030/`。']
(WORK/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');p=ROOT/'docs/TIMED_SPECTRUM_20261030.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/实际时间频谱实验_20261030.md');print(json.dumps(summary,indent=2))
