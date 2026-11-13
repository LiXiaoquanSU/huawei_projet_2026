"""Paired, record-isolated evaluation; all fits complete before exposing target labels."""
import csv
from run import *
PACKAGE=ROOT/'2026-10-25_037配方与种子复现'

def aggregate(files):
    d=[torch.load(f,weights_only=True) for f in files]
    return torch.stack([x['logits'].softmax(-1) for x in d]).mean(0).clamp_min(1e-12).log()

def bootstrap(records,field='delta'):
    rng=np.random.default_rng(20261025);values=[]
    for _ in range(5000):
        total=0;net=0
        for room in ROOMS:
            rr=[r for r in records if r['room']==room]
            for i in rng.integers(0,len(rr),size=len(rr)):
                net+=rr[i][field];total+=rr[i]['windows']
        values.append(net/total)
    return [float(x) for x in np.percentile(values,[2.5,97.5])]

def main():
    assert (WORK/'training_complete.json').exists();pro=read(WORK/'protocols.json');results=[];protocolproof=[]
    fits=sorted((WORK/'fits').glob('*.pt'));assert len(fits)==48
    for room in ROOMS:
        for seed in SEEDS:
            group=[torch.load(f,weights_only=True) for f in fits if f.name.startswith(f'{room}_s{seed}_')]
            assert len(group)==6 and len({g['source_draws_sha256'] for g in group})==1
            for g in group:
                assert set(g['source_fit_records']).isdisjoint(g['source_calibration_records'])
                assert set(g['target_adaptation_records']).isdisjoint(g['target_test_records'])
                assert not set(g['source_fit_records']+g['source_calibration_records'])&set(pro[room]['target_records'])
    for method in ['global','conditioned','view_u','conditional_u']:
        item=dict(method=method,protocol='DG' if method in ['global','conditioned'] else 'targetU',rooms={},records=[])
        net=total=correct=basecorrect=matchednet=0
        for room in ROOMS:
            d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room);base=old.policy(d['raw'],b,T)
            source=aggregate([WORK/'fits'/f'{room}_s{seed}_conditioned_source.pt' for seed in SEEDS])
            pred=source.clone();foldmetrics=[]
            for fold,ff in enumerate(pro[room]['folds']):
                if method in ['global','conditioned']:
                    z=aggregate([WORK/'fits'/f'{room}_s{seed}_{method}_source.pt' for seed in SEEDS])
                else:z=aggregate([WORK/'fits'/f'{room}_s{seed}_{method}_fold{fold}.pt' for seed in SEEDS])
                ix=ff['test_indices'];pred[ix]=z[ix]
                foldmetrics.append(dict(fold=fold,test_records=ff['test_records'],metrics=old.metrics(z,d['labels'],base,ix)))
            ix=old.split(room,d['rows'])[2];m=old.metrics(pred,d['labels'],base,ix);bm=old.metrics(base,d['labels'],base,ix)
            matched=old.metrics(pred,d['labels'],source,ix)
            item['rooms'][room]=dict(metrics=m,baseline=bm,matched_conditioned=matched,folds=foldmetrics)
            net+=m['correct']-bm['correct'];total+=len(ix);correct+=m['correct'];basecorrect+=bm['correct']
            matchednet+=matched['corrected']-matched['broken']
            for identifier in pro[room]['target_records']:
                jj=[i for i in ix if d['rows'][i]['identifier']==identifier];yy=d['labels'][jj]
                nc=int((pred[jj].argmax(-1)==yy).sum());bc=int((base[jj].argmax(-1)==yy).sum());sc=int((source[jj].argmax(-1)==yy).sum())
                item['records'].append(dict(room=room,identifier=identifier,windows=len(jj),correct=nc,baseline_correct=bc,delta=nc-bc,matched_delta=nc-sc))
            torch.save(dict(logits=pred,baseline=base,source_conditioned=source,labels=d['labels'],indices=ix,rows=d['rows']),WORK/f'eval_{method}_{room}.pt')
        item.update(net=net,total=total,correct=correct,baseline_correct=basecorrect,gain=net/total,
          room_gains={room:(item['rooms'][room]['metrics']['correct']-item['rooms'][room]['baseline']['correct'])/item['rooms'][room]['metrics']['total'] for room in ROOMS},
          matched_source_net=matchednet,ci95=bootstrap(item['records']),matched_source_ci95=bootstrap(item['records'],'matched_delta'))
        results.append(item)
    # Post-hoc diagnostic only: these target labels never entered adaptation or checkpoint choice.
    pseudo=[]
    for room in ROOMS:
        d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room)
        ix=old.split(room,d['rows'])[2];teacher=old.policy(d['raw'][ix],b,T).softmax(-1)
        views=(d['raw'][ix]/T+b).softmax(-1);win=teacher.argmax(-1)
        keep=(teacher.amax(-1)>=.8)&((views.argmax(-1)==win[:,None,None]).sum((1,2))>=10)
        selected=int(keep.sum());correct=int(((win==d['labels'][ix])&keep).sum())
        pseudo.append(dict(room=room,windows=len(ix),selected=selected,correct=correct,posthoc_accuracy=correct/selected if selected else None,
                           note='Target labels revealed only for post-training diagnosis, not changing threshold or model selection.'))
    save(WORK/'results.json',results);save(WORK/'pseudo_label_diagnostic.json',pseudo)
    dg=next(r for r in results if r['method']=='conditioned');accepted=[]
    for r in results:
        if r['protocol']!='targetU':continue
        passes=lambda q:q['gain']>=.02 and min(q['room_gains'].values())>=-.02 and sum(g>0 for g in q['room_gains'].values())>=2 and q['ci95'][0]>0
        # Claim U benefit only if it beats the matched source-conditioned control.
        r['candidate_gate_passed']=passes(dg) and passes(r) and r['matched_source_net']>0 and r['matched_source_ci95'][0]>0
        if r['candidate_gate_passed']:accepted.append(r)
    save(WORK/'results.json',results);save(WORK/'selection.json',dict(selected=max(accepted,key=lambda q:q['net'])['method'] if accepted else None,
        reason='Both protocols improve and U has positive paired record CI vs matched source-only control. No target-label checkpoint selection.',platform_score=None))
    lines=['# 环境适应研究：两套验证与48次小模块训练','','最高已测仍0.54，最近两个不同候选均0.53。本轮先检验验证协议与环境适应，不以本地提升承诺平台0.60。',
      '', '## 实验边界', '',
      '- 原054及四个历史留房间教师的主干/读出冻结；使用实际ONNX预处理对应的四路证据。条件模块584参数，全局仿射控制8参数；每路最多±10%缩放、±0.1源特征标准差偏移，没有类别偏置或房间ID输入。',
      '- 四目标房间×两种子：源全局/源条件各8次，目标U一致性/条件伪标签各16次，共48次。所有方法每轮有标签抽样完全相同，SHA已核验。',
      '- 目标房间整采集记录哈希分两组，适应/测试互斥，再交换；训练对象不含目标人数标签。源校准NLL只选择50/150步，目标标签在训练完成后才用于评估。',
      '- DG完全没有目标房间的训练/U记录；targetU模型仅见该房间适应组的无标签输入。两协议评估相同的目标记录，按记录配对统计。',
      '- 已知带标签房间也是不同layout；没有同layout两带标签房间。b1仅1记录不能拆适应/测试；unknown不合并为一个房间。历史代理不使用unknown源记录，真实全量054仍由全部48条训练。',
      '- 开发房间已反复使用，条件伪标签方案与一致性方案若相同可能因为可信样本不足；不把一致性当准确率。',
      '', '## 结果', '', '|协议/方法|正确/880|相对原模型净增|a1|c1|g1|e1|整记录95%提升区间|相对源条件净增|',
      '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in results:
        lo,hi=r['ci95'];g=[f'{r["room_gains"][room]*100:+.2f}' for room in ROOMS]
        lines.append('|'+ '|'.join([r['protocol']+'/'+r['method'],str(r['correct'])+'/'+str(r['total']),str(r['net']),*g,f'{lo*100:+.2f}～{hi*100:+.2f}',str(r['matched_source_net'])])+'|')
    lines+=['','单位：房间与区间为百分点。原教师对照278/880，并非平台0.54直接换算。',
      '', '## 伪标签事后诊断', '', '以下目标标签只在全部训练结束后用于解释失败，不用于调整阈值或重选检查点。']
    for r in pseudo:lines.append(f'- {r["room"]}：阈值选中{r["selected"]}/{r["windows"]}，事后正确{r["correct"]}。')
    audit=read(WORK/'data_audit.json')
    lines+=['','## 官方数据新审计', '',
      '- 198条ATI均重新扫描。S记录中位间隔范围5084～5119微秒，U为5083～5095微秒，约196Hz；没有发现负间隔。S与U均存在>20ms间隔，不能假定匀采样。',
      '- S整记录时长61.46～70.85秒，U为61.83～65.55秒。不能由一条中位采样率断定每个600窗都恰好3秒。',
      '- 原始CSI只抽查11条U记录的前600帧（一条/已知房间及全部5条unknown），不是全150条CSI审计；全部shape/dtype符合约定。精细统计见D盘data_audit.json。未复用历史FLOAT64预处理特征。',
      '', '## 决策', '',
      '通过完整候选门槛：'+(','.join(r['method'] for r in accepted) if accepted else '无。尚未证明环境条件模块和目标U在这套协议中有稳定净收益，不导出伪升级模型。'),
      '这不证明U无用；结论限于584参数的有界条件仿射、所用一致性/置信度规则和当前偏斜数据。后续可检验跨记录背景建模/条件表示，而不是把相同损失换名重跑。',
      '', '协议、完整记录划分、数据支持度、48份模块权重、原始输出和结果位于D:/huawei2026-experiments/environment_protocol_20261025/。代码位于experiments/environment_protocol_20261025/；支持度表docs/ENVIRONMENT_PROTOCOL_SUPPORT_20261025.csv。没有读取隐藏测试标签或提交平台。']
    report='\n'.join(lines)+'\n';(ROOT/'docs/ENVIRONMENT_PROTOCOL_20261025.md').write_text(report,encoding='utf-8')
    (PACKAGE/'环境适应双验证研究.md').write_text(report,encoding='utf-8')
    print(json.dumps([dict(method=r['method'],net=r['net'],rooms=r['room_gains'],ci=r['ci95'],relative_source=r['matched_source_net']) for r in results],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
