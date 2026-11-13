import json,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/readout_audit_20261030')
BASE=Path('D:/huawei2026-experiments/texture_chain_20261028')
r=json.loads((WORK/'results.json').read_text());p=json.loads((WORK/'predictions.json').read_text())
assert len(r['results'])==12
lines=['# 冻结表征与分类层诊断（2026-10-30）','','问题：训练后期下降，是否主要因为最终分类层过拟合，而骨干仍保留更好的可迁移人数信息？','','共6份已有权重、两层表征、4个正则强度，共48次逻辑回归拟合；没有重训骨干。标准化仅拟合源训练记录，正则强度仅用源校准均衡NLL选择。权重采用人数→房间→记录→窗口均衡；两时间视角表征取平均。各组完整原模型推理已复现误差小于1e-5。','','|权重|表征|正确/960|原分类层|相对原分类层差值95%区间|','|---|---|---:|---:|---|']
summary={};rng=np.random.default_rng(30)
for cp in ('selected','last'):
 for layer in ('prehead','penultimate'):
  rr=[v for v in r['results'] if v['checkpoint']==cp and v['layer']==layer];groups=[]
  for fold in range(3):
   d=torch.load(BASE/f'record{fold}_full_s42/predictions.pt',weights_only=False,map_location='cpu');ix=np.array(d['test_indices']);y=d['labels'].numpy()[ix]
   pred=np.array(p[f'record{fold}_{cp}_{layer}']['probabilities']).argmax(1)
   old=d['logits' if cp=='selected' else 'last_logits'][ix].argmax(1).numpy()
   ids=np.array([d['rows'][i]['identifier'] for i in ix]);delta=(pred==y).astype(float)-(old==y).astype(float)
   groups.append(np.array([delta[ids==i].mean() for i in sorted(set(ids))]))
  draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)]
  ci=np.percentile(draws,[2.5,97.5]);correct=sum(v['correct'] for v in rr);baseline=sum(v['original_correct'] for v in rr)
  summary[f'{cp}_{layer}']=dict(correct=correct,total=960,baseline=baseline,delta=(correct-baseline)/960,ci95=ci.tolist())
  lines.append(f'|{cp}|{layer}|{correct}/960 ({correct/960:.2%})|{baseline}/960|{ci[0]:+.2%} 至 {ci[1]:+.2%}|')
lines+=['','区间按每折整记录配对重采样3000次；未覆盖多次研究选择和种子不确定性。这些开发记录已反复使用，不是新的盲测。','','## 结论边界','','简单分类层能否改善，只能诊断可线性读取的信息；即使改善，也不能说明后期骨干更好或已消除房间捷径。未达到0.6不能用训练正确率、记录对排序率替代。没有导出新提交模型；平台最佳仍为已测0.54。','','代码：`experiments/readout_audit_20261030/run.py`；原始结果：`D:/huawei2026-experiments/readout_audit_20261030/`。']
(WORK/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
target=ROOT/'docs/READOUT_AUDIT_20261030.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile(target,ROOT/'2026-10-25_037配方与种子复现/冻结表征诊断_20261030.md')
print(json.dumps(summary,indent=2))
