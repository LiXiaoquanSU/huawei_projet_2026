import json,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];WORK=Path('D:/huawei2026-experiments/count_modes_20261030');BASE=Path('D:/huawei2026-experiments/texture_chain_20261028')
r=json.loads((WORK/'results.json').read_text())['results'];assert len(r)==12
lines=['# 每个人数的多模式假设（2026-10-30）','','前轮真实时间频谱在留房间验证失败，因此本轮检验另一假设：同人数可能对应不同的信号状态，单中心类别读出是否限制表现。冻结各折源训练的早期纹理骨干1664维特征；共同128维非线性映射后归一化，与每类1或4个可学习中心做余弦相似度、温度0.1的logsumexp融合。中心只代表潜在信号模式，不是受监督动作或独立人体。','','3折×2模式数×2种子=12组，60轮/组，共720轮小型读出训练。仅源训练数据拟合标准化、映射与中心；仅源校准均衡NLL选轮；固定60轮预测保留。四中心额外1920个参数；两种子只改变读出，不重训骨干。开发折反复使用，不能代替独立盲测。','','|每类中心数|种子|源选正确/960|固定60轮|源选轮数|','|---|---|---:|---:|---|']
summary={};rng=np.random.default_rng(30)
for k in (1,4):
 for seed in (42,2026):
  rr=[v for v in r if v['k']==k and v['seed']==seed];a=sum(v['test']['correct'] for v in rr);b=sum(v['last_test']['correct'] for v in rr);epochs=[v['epoch'] for v in rr];summary[f'k{k}_s{seed}']=dict(correct=a,total=960,last=b,epochs=epochs,occupancy=[v['fit_mode_occupancy'] for v in rr]);lines.append(f'|{k}|{seed}|{a}/960 ({a/960:.2%})|{b}/960|{epochs}|')
lines+=['','## 四中心相对单中心的配对差值','']
for seed in (42,2026):
 for checkpoint in ('logits','last_logits'):
  groups=[]
  for fold in range(3):
   d=torch.load(BASE/f'record{fold}_full_s42/predictions.pt',weights_only=False,map_location='cpu');a=torch.load(WORK/f'record{fold}_k4_s{seed}_predictions.pt',weights_only=False,map_location='cpu');b=torch.load(WORK/f'record{fold}_k1_s{seed}_predictions.pt',weights_only=False,map_location='cpu');ix=a['test_indices'];y=d['labels'][ix];delta=((a[checkpoint][ix].argmax(1)==y).float()-(b[checkpoint][ix].argmax(1)==y).float()).numpy();ids=np.array([d['rows'][i]['identifier'] for i in ix]);groups.append(np.array([delta[ids==k].mean() for k in sorted(set(ids))]))
  draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)];ci=np.percentile(draws,[2.5,97.5]);delta=float(np.concatenate(groups).mean());summary[f'difference_s{seed}_{checkpoint}']=dict(delta=delta,ci95=ci.tolist());lines.append(f'- 种子{seed}，{checkpoint}：{delta:+.2%}，整记录bootstrap95%区间 {ci[0]:+.2%} 至 {ci[1]:+.2%}。')
lines+=['','区间不覆盖多次开发选择和骨干种子不确定性。逐类中心使用数量已写入summary，避免把塌缩成单一中心的网络称作实现了多状态识别。没有全量训练、ONNX导出或新平台成绩。平台最佳仍0.54，目标0.6未达到。']
(WORK/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');p=ROOT/'docs/COUNT_MODES_20261030.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/人数多模式读出_20261030.md');print(json.dumps({k:v for k,v in summary.items() if 'occupancy' not in v},indent=2));print([(k,v['correct']) for k,v in summary.items() if 'correct' in v])
