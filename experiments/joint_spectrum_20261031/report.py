import json,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];WORK=Path('D:/huawei2026-experiments/joint_spectrum_20261031')
lines=['# 纹理与实际时间频谱联合学习（2026-10-31）','','问题：独立频谱模型丢失静态/时间顺序信息，是否只有与原纹理联合学习才有价值？从零初始化训练同一纹理主干，接4×7×32频谱的两层卷积，频谱输出128维与原1664维晚融合。对照拥有完全相同参数与初始化，但频谱输入全零。零分支的有效容量不同，不能把同参数量等同于同有效容量。','','三折整记录、种子42、两方案、每组12轮，共6组72轮。源校准均衡NLL选轮，固定12轮保留。相同AdamW、采样、时间反转一致性；无旧平台权重、无外部/无标签数据。只用源记录训练，测试记录无交叠。开发折已重复使用，尚不是独立盲测。','','|方案|权重|正确/960|各折正确数|选择轮数|','|---|---|---:|---|---|']
summary={};rng=np.random.default_rng(31)
for variant in ('full','zero'):
 for cp in ('selected','last'):
  rr=[json.loads((WORK/f'record{i}_{variant}_s42/result.json').read_text()) for i in range(3)];key='test' if cp=='selected' else 'last_test';counts=[v[key]['correct'] for v in rr];epochs=[v['epoch'] for v in rr] if cp=='selected' else [12]*3;summary[f'{variant}_{cp}']=dict(correct=sum(counts),total=960,counts=counts,epochs=epochs,parameters=rr[0]['parameters'],confusion=np.sum([v[key]['confusion'] for v in rr],0).tolist());lines.append(f'|{variant}|{cp}|{sum(counts)}/960 ({sum(counts)/960:.2%})|{counts}|{epochs}|')
for cp in ('selected','last'):
 groups=[]
 for i in range(3):
  a=torch.load(WORK/f'record{i}_full_s42/predictions.pt',weights_only=False,map_location='cpu');b=torch.load(WORK/f'record{i}_zero_s42/predictions.pt',weights_only=False,map_location='cpu');ix=a['test_indices'];key='logits' if cp=='selected' else 'last_logits';y=a['labels'][ix];delta=((a[key][ix].argmax(1)==y).float()-(b[key][ix].argmax(1)==y).float()).numpy();ids=np.array([a['rows'][j]['identifier'] for j in ix]);groups.append(np.array([delta[ids==k].mean() for k in sorted(set(ids))]))
 draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)];ci=np.percentile(draws,[2.5,97.5]);delta=float(np.concatenate(groups).mean());summary[f'difference_{cp}']=dict(delta=delta,ci95=ci.tolist());lines+=['',f'{cp}相对置零：{delta:+.2%}；按整记录重采样3000次的95%区间 {ci[0]:+.2%} 至 {ci[1]:+.2%}。']
lines+=['','尚未执行新增种子、留房间、全量训练和ONNX导出。没有新平台分数；已测最佳0.54保留。不能把本地拟合分数解释为平台突破。原始权重与逐窗口预测：`D:/huawei2026-experiments/joint_spectrum_20261031/`。']
(WORK/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');p=ROOT/'docs/JOINT_SPECTRUM_20261031.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/纹理频谱联合实验_20261031.md');print(json.dumps(summary,indent=2))
