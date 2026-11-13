import sys,json,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'));import g60_common as old
WORK=Path('D:/huawei2026-experiments/pair_views_20261102');CACHE=Path('D:/huawei2026-experiments/g60_20261025')
r=old.read(WORK/'results.json')['results'];assert len(r)==32
lines=['# 借鉴054的候选人数比较实验（2026-11-02）','','已证实的054经验来自原权重不变时的条件多视角融合，不能据此推断扩大网络有效。本轮用类共享的两两比较器读取12视角证据，比较结果强制反对称，再汇总为每类±0.5以内修正。没有人数ID/房间ID输入；保留保护规则与允许修正全部输出各设一组，并以将12视角压成相同均值的同参数对照检查新增视角信息。','','4个留房间代理×4方案×2种子=32次训练，每次200步。教师核验fit权重哈希；沿用其源校准偏置/温度；修正器仅用源fit记录训练，仅源校准选0/20/50/100/200步。实际054全量权重不用于留房间验证。旧开发房间已反复使用，不能把小幅改善视为独立泛化证明。','','|方案|种子|正确/880|净变化|各房间净变化|选择步数|','|---|---|---:|---:|---|---|']
summary={}
for variant in ('views_protected','mean_protected','views_all','mean_all'):
 for seed in (42,2026):
  rr=[v for v in r if v['variant']==variant and v['seed']==seed];correct=sum(v['test']['correct'] for v in rr);delta=[v['test']['corrected']-v['test']['broken'] for v in rr];summary[f'{variant}_s{seed}']=dict(correct=correct,total=880,net=sum(delta),room_net=delta,steps=[v['step'] for v in rr]);lines.append(f'|{variant}|{seed}|{correct}/880 ({correct/880:.2%})|{sum(delta):+d}|{delta}|{[v["step"] for v in rr]}|')
audit=[]
for room in ('a1','c1','g1','e1'):
 d=torch.load(CACHE/f'{room}.pt',weights_only=True);b,T=old.calibration(room);ix=old.split(room,d['rows'])[2];y=d['labels'][ix];raw=d['raw'][ix];base=old.policy(raw,b,T);p=(raw/T+b+torch.tensor([0,.35,0,0,0])).reshape(-1,12,5);pred=p.argmax(-1);any_correct=(pred==y[:,None]).any(1);top2=(p.topk(2,-1).indices==y[:,None,None]).any(-1).any(-1);wrong=base.argmax(-1)!=y
 audit.append(dict(room=room,total=len(ix),base_correct=int((~wrong).sum()),wrong=int(wrong.sum()),wrong_with_any_correct_view=int((wrong&any_correct).sum()),wrong_without_correct_view=int((wrong&~any_correct).sum()),wrong_without_true_top2_any_view=int((wrong&~top2).sum()),any_correct_view=int(any_correct.sum()),by_class=[dict(label=c,total=int((y==c).sum()),wrong=int((wrong&(y==c)).sum()),wrong_without_correct_view=int((wrong&~any_correct&(y==c)).sum())) for c in range(5)]))
lines+=['','## 既有视角的信息支持诊断','','这些指标使用真值，仅作错误分析，不能部署、不能当模型正确率。任意视角正确只是直接选某个视角的理想上限，不是所有融合或新模型的上限。','','|房间|基线错误|错误中至少一视角正确|所有视角均错|所有视角前二均无真值|','|---|---:|---:|---:|---:|']
for a in audit:lines.append(f'|{a["room"]}|{a["wrong"]}|{a["wrong_with_any_correct_view"]}|{a["wrong_without_correct_view"]}|{a["wrong_without_true_top2_any_view"]}|')
lines+=['','平台最好仍0.54；大版已测0.25。此轮没有新全量模型或ONNX，不将训练代理分数与平台分数混淆。']
old.save(WORK/'summary.json',dict(results=summary,view_support=audit));p=ROOT/'docs/PAIR_VIEWS_20261102.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/054视角候选比较_20261102.md');print(summary);print(audit)
