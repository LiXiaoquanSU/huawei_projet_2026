import json,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];WORK=Path('D:/huawei2026-experiments/distribution_pool_20261030');BASE=Path('D:/huawei2026-experiments/texture_chain_20261028')
r=json.loads((WORK/'results.json').read_text())['results'];p=json.loads((WORK/'predictions.json').read_text());assert len(r)==12
lines=['# 局部分布池化机制实验（2026-10-30）','','假设：局部纹理中的多种信号模式，可能在均值/标准差汇总时丢失。每折冻结同一份早期CNN，提取池化前局部256维特征，以固定正交投影降至32维；只在源训练记录中拟合标准化与8个局部模式原型。按频率位置统计软分配比例与残差，接简单五类分类器。原型是信号模式，不能解释为独立人体。','','三折、四种输入、四个正则强度，共48次拟合。正则强度仅按源校准均衡NLL选择；所有预处理、原型及分类器只拟合源训练记录。标准化均值/方差控制与分布池化使用相同局部特征。旧1664维表征为另一基线。分布版特征维数更高，并非严格等容量；改善后还需容量控制。','','|输入|正确/960|每折正确数|相对局部均值/标准差95%区间|','|---|---:|---|---|']
summary={};rng=np.random.default_rng(30)
for mode in ('old','moments','distribution','old_plus_distribution'):
 rr=[v for v in r if v['mode']==mode];groups=[]
 for fold in range(3):
  d=torch.load(BASE/f'record{fold}_full_s42/predictions.pt',weights_only=False,map_location='cpu');ix=d['test_indices'];y=d['labels'][ix].numpy();pred=np.array(p[f'record{fold}_{mode}']['probabilities']).argmax(1);base=np.array(p[f'record{fold}_moments']['probabilities']).argmax(1);delta=(pred==y).astype(float)-(base==y).astype(float);ids=np.array([d['rows'][i]['identifier'] for i in ix]);groups.append(np.array([delta[ids==k].mean() for k in sorted(set(ids))]))
 draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)];ci=np.percentile(draws,[2.5,97.5]);counts=[v['correct'] for v in rr];summary[mode]=dict(correct=sum(counts),total=960,counts=counts,dimensions=rr[0]['dimensions'],delta_vs_moments=float(np.concatenate(groups).mean()),ci95=ci.tolist(),confusion=np.sum([v['confusion'] for v in rr],0).tolist());lines.append(f'|{mode}|{sum(counts)}/960 ({sum(counts)/960:.2%})|{counts}|{ci[0]:+.2%} 至 {ci[1]:+.2%}|')
lines+=['','整记录配对bootstrap3000次，不覆盖种子与多次结构选择不确定性。此为反复使用的开发集，不是独立盲测，也不是平台成绩。没有使用测试记录拟合原型或校准。','','模型原型、投影、分类器参数、特征和预测已保存在`D:/huawei2026-experiments/distribution_pool_20261030/`；未进行全数据训练/ONNX导出，平台最佳仍0.54。']
(WORK/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');target=ROOT/'docs/DISTRIBUTION_POOL_20261030.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(target,ROOT/'2026-10-25_037配方与种子复现/局部分布池化_20261030.md');print(json.dumps(summary,indent=2))
