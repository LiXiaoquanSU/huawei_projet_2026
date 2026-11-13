import json,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];WORK=Path('D:/huawei2026-experiments/preserve_features_20261030');BASE=Path('D:/huawei2026-experiments/texture_chain_20261028')
lines=['# 保留早期表征对照（2026-10-30）','','9组继续训练，各8轮，共72轮。三个整记录开发折；源校准选出的早期权重作为相同起点。所有方案同学习率、优化器、采样、初始化和种子。joint正常继续，frozen仅训练整个分类头，anchored联合训练并约束1664维融合特征偏离初始值。推理结构均不增加参数。','','每组允许源校准选择不更新的epoch0，并报告固定第8轮；校准只用源记录，目标记录不参与拟合、特征锚定或选择。旧校准集再次使用会产生选择偏差，本实验不是独立盲测。','','|方案|权重|正确/960|各折正确数|选择轮数|相对原早期权重差值95%区间|','|---|---|---:|---|---|---|']
summary={};rng=np.random.default_rng(30)
for variant in ('joint','frozen','anchored'):
 for cp in ('selected','last'):
  rr=[];groups=[];cm=np.zeros((5,5),dtype=int)
  for fold in range(3):
   folder=WORK/f'record{fold}_{variant}_s42';r=json.loads((folder/'result.json').read_text());rr.append(r)
   d=torch.load(folder/'predictions.pt',weights_only=False,map_location='cpu');b=torch.load(BASE/f'record{fold}_full_s42/predictions.pt',weights_only=False,map_location='cpu')
   ix=d['test_indices'];y=d['labels'][ix].numpy();p=d['logits' if cp=='selected' else 'last_logits'][ix].argmax(1).numpy();base=b['logits'][ix].argmax(1).numpy()
   ids=np.array([d['rows'][i]['identifier'] for i in ix]);delta=(p==y).astype(float)-(base==y).astype(float);groups.append(np.array([delta[ids==k].mean() for k in sorted(set(ids))]));cm+=np.bincount(y*5+p,minlength=25).reshape(5,5)
  draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)];ci=np.percentile(draws,[2.5,97.5]);key='test' if cp=='selected' else 'last_test';counts=[v[key]['correct'] for v in rr];epochs=[v['epoch'] for v in rr] if cp=='selected' else [8]*3
  summary[f'{variant}_{cp}']=dict(correct=sum(counts),total=960,counts=counts,epochs=epochs,ci95=ci.tolist(),confusion=cm.tolist())
  lines.append(f'|{variant}|{cp}|{sum(counts)}/960 ({sum(counts)/960:.2%})|{counts}|{epochs}|{ci[0]:+.2%} 至 {ci[1]:+.2%}|')
lines+=['','原早期纹理权重为479/960（49.90%），原始连续训练第20轮388/960（40.42%）。本轮重置了优化器和8轮调度，不能将差值全部归因于冻结/锚定；应先比较本轮三组匹配对照。区间按每折整记录重采样3000次，未涵盖结构选择和种子不确定性。','','## 相对本轮正常继续训练的匹配对照','']
for variant in ('frozen','anchored'):
 for cp in ('selected','last'):
  groups=[]
  for fold in range(3):
   d=torch.load(WORK/f'record{fold}_{variant}_s42/predictions.pt',weights_only=False,map_location='cpu');b=torch.load(WORK/f'record{fold}_joint_s42/predictions.pt',weights_only=False,map_location='cpu');ix=d['test_indices'];key='logits' if cp=='selected' else 'last_logits';y=d['labels'][ix];delta=((d[key][ix].argmax(1)==y).float()-(b[key][ix].argmax(1)==y).float()).numpy();ids=np.array([d['rows'][i]['identifier'] for i in ix]);groups.append(np.array([delta[ids==k].mean() for k in sorted(set(ids))]))
  draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)];ci=np.percentile(draws,[2.5,97.5]);delta=float(np.concatenate(groups).mean());summary[f'{variant}_minus_joint_{cp}']=dict(delta=delta,ci95=ci.tolist());lines.append(f'- {variant}/{cp}：{delta:+.2%}，95%区间 {ci[0]:+.2%} 至 {ci[1]:+.2%}。')
lines+=['','所有权重、预测及训练漂移记录保存在`D:/huawei2026-experiments/preserve_features_20261030/`。未进行全量训练或ONNX导出，平台最佳仍0.54。']
(WORK/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');p=ROOT/'docs/PRESERVE_FEATURES_20261030.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/保留早期表征_20261030.md');print(json.dumps(summary,indent=2))
