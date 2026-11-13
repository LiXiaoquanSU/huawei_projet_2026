import sys,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'));import g60_common as old
WORK=Path('D:/huawei2026-experiments/aligned_views_20261102');ROWS=Path('D:/huawei2026-experiments/next053_20261025/rows.json');rows=old.read(ROWS);y=torch.tensor([v['headcount'] for v in rows]);rr=old.read(WORK/'results.json')['results'];assert len(rr)==12
lines=['# 早期跨通道联合卷积与多视角训练（2026-11-02）','','借鉴054的可验证特点：联合处理接收通道，以及排列/时间视角的互补。新网络从零训练，将静态幅度3、动态幅度3、相位6通道对齐输入共享残差CNN；保留4个频率区域的均值/标准差/最大值及13维摘要，五类读出。相位32频组重复到128只是对齐，不恢复频率细节。没有复用全数据037/054权重。','','对照：同初始权重及训练预算，仅改变是否对训练批次随机使用6种一致Rx/相位对排列。训练均使用正反时间两视图，NLL+.05时间一致性。推理分别使用原顺序双时间视图与全部6Rx×2时间视图。各自仅源校准均衡NLL选择轮数，同时保留固定末轮。3折×2训练条件，每组12轮，共6次训练72轮、12组读出评估。','','|训练|推理Rx视角|源选正确/960|固定12轮|源选轮数|','|---|---|---:|---:|---|']
summary={};rng=np.random.default_rng(2)
for variant in ('identity_train','permuted_train'):
 for views in (1,6):
  group=[r for r in rr if r['variant']==variant and r['views']==views];a=sum(r['test']['correct'] for r in group);b=sum(r['last_test']['correct'] for r in group);epochs=[r['epoch'] for r in group];summary[f'{variant}_v{views}']=dict(correct=a,total=960,last=b,epochs=epochs,parameters=group[0]['parameters'],confusion=np.sum([r['test']['confusion'] for r in group],0).tolist());lines.append(f'|{variant}|{views}|{a}/960 ({a/960:.2%})|{b}/960|{epochs}|')
lines+=['','## 整记录匹配差值','']
for new,base in [('permuted_train_v1','identity_train_v1'),('permuted_train_v6','identity_train_v6'),('identity_train_v6','identity_train_v1'),('permuted_train_v6','permuted_train_v1')]:
 groups=[]
 for fold in range(3):
  a=torch.load(WORK/f'record{fold}_{new}_predictions.pt',weights_only=False,map_location='cpu');b=torch.load(WORK/f'record{fold}_{base}_predictions.pt',weights_only=False,map_location='cpu');ix=a['test_indices'];delta=((a['logits'][ix].argmax(1)==y[ix]).float()-(b['logits'][ix].argmax(1)==y[ix]).float()).numpy();ids=np.array([rows[i]['identifier'] for i in ix]);groups.append(np.array([delta[ids==k].mean() for k in sorted(set(ids))]))
 draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)];ci=np.percentile(draws,[2.5,97.5]);delta=float(np.concatenate(groups).mean());summary[f'{new}_minus_{base}']=dict(delta=delta,ci95=ci.tolist());lines.append(f'- {new}−{base}：{delta:+.2%}，95%区间 {ci[0]:+.2%} 至 {ci[1]:+.2%}。')
lines+=['','本轮没有同容量“先独立编码后融合”的结构对照，因此不能把与旧模型的差异单独归因于早期通道交互；能直接比较的是排列增广和推理融合。只有一个训练种子、重复开发划分；未完成留房间、全量训练与ONNX导出。平台最好仍0.54，大版0.25仍保留失败记录。']
old.save(WORK/'summary.json',summary);p=ROOT/'docs/ALIGNED_VIEWS_20261102.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/跨通道联合与视角训练_20261102.md');print(summary)
