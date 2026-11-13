import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];W=Path('D:/huawei2026-experiments/dynamic_evidence_20261108')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 r=read(W/'results.json');within=read(W/'within_room.json');assert len(r)==9 and len(within['results'])==18;rows=read(Path('D:/huawei2026-experiments/next053_20261025/rows.json'));aggregate={};comp=[]
 for mode in ['raw','static','dynamic']:
  rr=[x for x in r if x['mode']==mode];cm=np.array([x['metrics']['confusion'] for x in rr]).sum(0);aggregate[mode]=dict(correct=int(np.trace(cm)),total=960,per_fold=[x['metrics']['correct'] for x in rr],class_recall=(np.diag(cm)/cm.sum(1)).tolist(),confusion=cm.tolist(),source_correct=sum(x['source_fit']['correct'] for x in rr),source_total=sum(x['source_fit']['total'] for x in rr))
 deltas=[]
 for f in range(3):
  a=torch.load(W/f'record{f}_static/predictions.pt',weights_only=False,map_location='cpu');b=torch.load(W/f'record{f}_dynamic/predictions.pt',weights_only=False,map_location='cpu');assert a['indices']==b['indices'] and torch.equal(a['labels'],b['labels']);y=a['labels'];p=a['logits'].argmax(-1)==y;q=b['logits'].argmax(-1)==y;comp.append(dict(fold=f,static_only=int((p&~q).sum()),dynamic_only=int((~p&q).sum()),both_correct=int((p&q).sum()),both_wrong=int((~p&~q).sum()),oracle=int((p|q).sum()),equal_probability=int(((a['logits'].exp()+b['logits'].exp()).argmax(-1)==y).sum())))
  diff=(q.float()-p.float()).numpy();idx=a['indices'];ids=sorted({rows[i]['identifier'] for i in idx});deltas.append(np.array([diff[[k for k,i in enumerate(idx) if rows[i]['identifier']==ident]].mean() for ident in ids]))
 rng=np.random.default_rng(20261108);draw=np.array([np.concatenate([rng.choice(d,len(d),replace=True) for d in deltas]).mean() for _ in range(10000)]);ci=np.quantile(draw,[.025,.975]).tolist()
 within_summary=[]
 for room in ['a1','c1','g1']:
  for mode in ['raw','static','dynamic']:
   rr=[x for x in within['results'] if x['room']==room and x['mode']==mode];cm=np.array([x['confusion'] for x in rr]).sum(0);within_summary.append(dict(room=room,mode=mode,classes=rr[0]['classes'],correct=int(np.trace(cm)),total=int(cm.sum()),macro_recall=float((np.diag(cm)/cm.sum(1)).mean()),per_fold=[dict(correct=x['correct'],total=x['total']) for x in rr]))
 out=dict(date='2026-11-08',neural_fits=9,epochs=108,linear_probes=18,aggregate=aggregate,complementarity=comp,dynamic_minus_static_record_bootstrap95=ci,within_room=within_summary,status='completed_no_export',platform_best=.54,platform_score=None,next_submission=None)
 save(Path(__file__).with_name('results_summary.json'),out)
 lines=['# 动态信息能否跨记录识别人：2026-11-08','','本轮完成9次神经训练（108轮）和18次同房间线性探针。所有结果均来自复用的本地记录，不是平台隐藏测试。已测最佳仍0.54；ordinal75已测0.47，不再推荐。没有新增提交文件，0.60目标未完成。','','## 一、匹配条件的三组网络','','复用原始CSI的有界联合幅相轨迹，比较完整轨迹raw、每条轨迹时间均值static、去均值后按天线对共享RMS归一化dynamic。三个结构参数一致，17维额外摘要全部置零；固定seed42、12轮、同一优化器和采样预算；外层完整记录不参与训练或选轮。static/dynamic是信号分解，不是静止人/移动人的语义标签。','','|输入|各折正确/320|正确/960|准确率|源拟合正确/1920|','|---|---|---:|---:|---:|']
 for mode,d in aggregate.items():lines.append(f"|{mode}|{d['per_fold']}|{d['correct']}|{d['correct']/960:.2%}|{d['source_correct']}|")
 lines+=['','源拟合1920次来自三个模型各自640个训练窗口，同一原始记录重复出现，既不是独立样本数，也不是泛化成绩。',f"动态相对静态净增14/960，但按整记录、折内重采样的描述性95%差值区间为[{ci[0]:+.2%},{ci[1]:+.2%}]。第一折提高40个，第二折下降3个，第三折下降23个；不是一致提升。区间不校正多轮研究选择偏差。",'','## 二、错误互补与简单融合','','|折|仅静态正确|仅动态正确|真值择优上限|等概率平均|','|---|---:|---:|---:|---:|']
 for c in comp:lines.append(f"|{c['fold']}|{c['static_only']}|{c['dynamic_only']}|{c['oracle']}|{c['equal_probability']}|")
 lines+=['','两个已训练模型由真值决定选哪个时，合计519/960；等概率平均为408/960。这种择优依赖真值，不能部署。它只是这两组当前决策的诊断上限，不是所有新模型的上限，也不能推断平台极限。缺少进一步收益依据，本轮不再增加门控或导出平均模型。','','## 三、已知同房间的诊断','','a1/c1只比较1—4人，g1只比较0/1；仅纳入每类至少两条记录的任务。各房间独立按完整记录二折，标准化只拟合源记录，逻辑回归固定C=0.01。房间与候选人数集合已知，任务比比赛五分类不同，不能用其中任一高分声称达到0.60。','','|房间|信号|正确/总数|准确率|类别平均召回|','|---|---|---:|---:|']
 for v in within_summary:lines.append(f"|{v['room']}|{v['mode']}|{v['correct']}/{v['total']}|{v['correct']/v['total']:.2%}|{v['macro_recall']:.2%}|")
 lines+=['','a1动态109/220，比静态93/220好；c1动态49/240，仍弱于四类均匀随机选择的25%期望准确率；g1动态190/340，但该子集全判人数较多的1人已有200/340。因此这里没有证据证明动态轨迹已稳定提供可迁移的人数信息。少量不同采集记录还可能包含布局、活动、设备状态差异，已知房间并不消除这些混杂。','','## 结论与后续边界','','去掉静态信息并归一化动态，确实改变了学到的表征，却没有带来可靠的人数泛化。既不能说动态信号没有人数信息，也不能假定静态信号都是噪声或应当删除。当前特征、模型、样本覆盖和优化仍可能不足。','下一步若重新建模，需保留静态占用信息，同时验证动态信息对具体人数混淆是否有独立增量；在没有该证据前，不把静态+动态相加或提高参数量当成合理改进已经成立。原054继续作为已评分参照；本轮停止这组分解与小网络扩容。','','程序：`experiments/dynamic_evidence_20261108/`；检查点、预测和完整协议：`D:/huawei2026-experiments/dynamic_evidence_20261108/`。']
 report=ROOT/'docs/DYNAMIC_EVIDENCE_20261108.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');reg=read(ROOT/'models/registry.json');reg['updated_date']='2026-11-08';reg['research_status']='dynamic_evidence_matched_controls_complete_no_export';reg['dynamic_evidence_20261108']=out;reg['next_submission']=None;save(ROOT/'models/registry.json',reg)
 p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');prefix='最新研究（2026-11-08）：完成9次静态/动态/完整轨迹同容量训练和18次同房间探针。动态389/960、静态375/960，跨折不稳定；同房间结果不支持稳定人数增量。未导出新模型，平台最佳仍0.54。见[动态证据报告](docs/DYNAMIC_EVIDENCE_20261108.md)。\n\n';p.write_text(prefix+text,encoding='utf-8');print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
