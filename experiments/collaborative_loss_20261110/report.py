import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];D=Path('D:/huawei2026-experiments');W=D/'collaborative_loss_20261110'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 rs=read(W/'results.json');assert len(rs)==12;agg={};contrasts={};rows=read(D/'next053_20261025/rows.json')
 for seed in [42,7]:
  for variant in ['control','no_expert_ce']:
   rr=[r for r in rs if r['seed']==seed and r['variant']==variant];assert len(rr)==3;cm=np.array([r['metrics']['confusion'] for r in rr]).sum(0);agg[f'{seed}_{variant}']=dict(correct=int(cm.trace()),total=int(cm.sum()),fold_correct=[r['metrics']['correct'] for r in rr],class_recall=(cm.diagonal()/cm.sum(1)).tolist(),confusion=cm.tolist(),source_correct=sum(r['source_fit']['correct'] for r in rr))
  diffs=[];fixed=broken=0
  for f in range(3):
   pp=[torch.load(W/f'record{f}_s{seed}_{v}'/'predictions.pt',weights_only=False,map_location='cpu') for v in ['control','no_expert_ce']];a,b=pp;assert a['draws']==b['draws'] and a['indices']==b['indices'] and torch.equal(a['labels'],b['labels']);pa=a['logits'].argmax(-1);pb=b['logits'].argmax(-1);y=a['labels'];fixed+=int(((pa!=y)&(pb==y)).sum());broken+=int(((pa==y)&(pb!=y)).sum());ids=np.array([rows[i]['identifier'] for i in a['indices']]);delta=((pb==y).float()-(pa==y).float()).numpy();diffs.append(np.array([delta[ids==ident].mean() for ident in sorted(set(ids))]))
  rng=np.random.default_rng(20261110);boot=[np.concatenate([rng.choice(d,len(d),replace=True) for d in diffs]).mean() for _ in range(10000)];contrasts[str(seed)]=dict(corrected=fixed,broken=broken,record_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
 out=dict(date='2026-11-10',fits=12,epochs=60,aggregate=agg,contrasts=contrasts,platform_score=None,limitations='Repeated development record folds; same records across seeds; no blind or platform accuracy; bootstrap does not correct researcher selection.')
 save(Path(__file__).with_name('results_summary.json'),out)
 lines=['# 分支分类损失对照（2026-11-10）','','## 设计','','保留原037的完整四路结构。对照原损失与仅去掉0.15×专家独立交叉熵；联合直接分类、有序头、JS与信号辅助损失不变。专家仍通过混合分类路径获得梯度。seed42/7各三折，整记录隔离，固定5轮AdamW；不按外层分数选轮或调偏置。初始化前设种子，保留初始化后随机流，使用原生DataLoader及double WeightedRandomSampler。各对照采样序列逐项相同。','','|种子/方案|各折正确/320|正确/960|训练拟合/1920|','|---|---|---:|---:|']
 for k,v in agg.items():lines.append(f"|{k}|{v['fold_correct']}|{v['correct']}|{v['source_correct']}|")
 lines+=['','## 成对变化','']
 for k,v in contrasts.items():lines.append(f"- seed{k}：纠正{v['corrected']}、破坏{v['broken']}；记录重采样描述性95%差值区间{v['record_bootstrap95']}。")
 lines+=['','相同记录跨种子重复出现，不能把1920验证预测当成1920独立窗口。全部是反复开发的切分，不是盲测。高训练拟合不能视为泛化。原平台0.54权重没有参与这些训练，不用其训练内成绩替代验证。','','## 关联诊断','','另见experiments/branch_audit_20261110/results.json及view_checks.json。比较原037与重训seed42/0/304，在每条S/U记录首窗上分析12视角；S为训练已见、U无标签。原权重的接收排列稳定性高于这三个重训权重，不能由此推断因果或真实人数正确性。U使用历史float64预处理缓存，S使用当前float32缓存；所有权重在各域接收同一输入，不把该审计称为精确部署复现。','','全部权重和预测保存在D:/huawei2026-experiments/collaborative_loss_20261110。']
 (ROOT/'docs/COLLABORATIVE_LOSS_20261110.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
