import json,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];WORK=Path('D:/huawei2026-experiments/context_spectrum_20261031');BASE=Path('D:/huawei2026-experiments/texture_chain_20261028')
r=json.loads((WORK/'results.json').read_text())['results'];assert len(r)==18
lines=['# 固定纹理骨干的上下文频谱修正（2026-10-31）','','根据联合从零训练的对照结果，检验固定已有特征是否能避免训练漂移并利用频谱。使用每折源训练得到的早期纹理骨干，其1664维表征和五类概率保持不变；新增纹理128维映射、频谱128维编码和二者tanh逐元素乘积，再经小读出产生每类±0.5以内的logit修正。初始化为原模型等价。','','三方案：product包含显式乘积；no_product将乘积槽置零但仍有两个分支，MLP仍可产生隐式交互；zero将频谱输入置零，保留相同分支和参数。3折×3方案×2种子，每组30轮，共18组540轮修正器训练。种子只重训修正器，骨干不变。允许源校准选回epoch0，固定30轮也报告。','','只用源训练记录拟合与缩放；反复使用的开发折不属于独立验证。同参数量不代表同有效容量。没有平台标签、无标签伪标签或旧全数据权重参与评估。','','|方案|种子|源选正确/960|固定30轮|选择轮数|','|---|---|---:|---:|---|']
summary={};rng=np.random.default_rng(31)
for variant in ('product','no_product','zero'):
 for seed in (42,2026):
  rr=[v for v in r if v['variant']==variant and v['seed']==seed];a=sum(v['test']['correct'] for v in rr);b=sum(v['last_test']['correct'] for v in rr);epochs=[v['epoch'] for v in rr];summary[f'{variant}_s{seed}']=dict(correct=a,total=960,last=b,epochs=epochs,confusion=np.sum([v['test']['confusion'] for v in rr],0).tolist());lines.append(f'|{variant}|{seed}|{a}/960 ({a/960:.2%})|{b}/960|{epochs}|')
lines+=['','原纹理骨干479/960（49.90%）。匹配对照差值：','']
for seed in (42,2026):
 for compare in ('zero','no_product'):
  groups=[]
  for fold in range(3):
   d=torch.load(BASE/f'record{fold}_full_s42/predictions.pt',weights_only=False,map_location='cpu');a=torch.load(WORK/f'record{fold}_product_s{seed}_predictions.pt',weights_only=False,map_location='cpu');b=torch.load(WORK/f'record{fold}_{compare}_s{seed}_predictions.pt',weights_only=False,map_location='cpu');ix=a['test_indices'];y=d['labels'][ix];delta=((a['logits'][ix].argmax(1)==y).float()-(b['logits'][ix].argmax(1)==y).float()).numpy();ids=np.array([d['rows'][i]['identifier'] for i in ix]);groups.append(np.array([delta[ids==k].mean() for k in sorted(set(ids))]))
  draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)];ci=np.percentile(draws,[2.5,97.5]);delta=float(np.concatenate(groups).mean());summary[f'product_minus_{compare}_s{seed}']=dict(delta=delta,ci95=ci.tolist());lines.append(f'- seed{seed}，product−{compare}：{delta:+.2%}，整记录bootstrap95%区间 {ci[0]:+.2%} 至 {ci[1]:+.2%}。')
lines+=['','不因某个种子或某一类改善就认定机制成立；区间未覆盖骨干种子和历史结构选择不确定性。对应留房间及全数据训练未执行，没有新ONNX或新平台成绩。平台最佳仍0.54，目标0.6未达到。完整权重与逐窗口预测：`D:/huawei2026-experiments/context_spectrum_20261031/`。']
(WORK/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');p=ROOT/'docs/CONTEXT_SPECTRUM_20261031.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/固定纹理上下文频谱_20261031.md');print([(k,v) for k,v in summary.items() if 'correct' not in v]);print([(k,v['correct'],v['last']) for k,v in summary.items() if 'correct' in v])
