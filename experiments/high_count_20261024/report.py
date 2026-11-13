"""Record .48 plateau and publish the independently checked high-count readout."""
import csv
import shutil
from pathlib import Path
from export_learned import ROOT,WORK,EXPORT,audit


def main():
    published=ROOT/'models/high_count_20261024/01_conditional_high_count.json'
    if published.exists() and audit.read(published).get('platform_score') is not None:
        print('Scored artifact already published; preserve user feedback and later candidate selection.');return
    digest=audit.sha(EXPORT/'model.onnx');validation=audit.read(EXPORT/'validation.json')
    resources=audit.read(EXPORT/'resource_check.json');independent=audit.read(EXPORT/'independent_reference_check.json')
    assert resources['status']=='pass' and resources['sha256']==validation['candidate_sha256']==digest
    assert independent['max_logit_error']<2e-5 and audit.sha(audit.SOURCE)==audit.SOURCE_HASH
    all_rows=audit.read(EXPORT/'all_predictions.json')
    relevant={r['identifier'] for r in all_rows if r['prediction'] in (3,4)}
    sources=audit.read(Path('D:/huawei2026-experiments/actual047_audit_20261024/summary.json'))['source_files']
    verified=[]
    for r in sources:
        if r['identifier'] not in relevant:continue
        for prefix,key in [('CSI','csi_sha256'),('ATI','ati_sha256')]:
            path=Path('D:/huawei2026-data/labeled')/f"{prefix}_{r['identifier']}.b2"
            assert audit.sha(path)==r[key]
        verified.append(r)
    assert len(verified)==len(relevant)
    audit.save(EXPORT/'raw_source_proof.json',dict(verified_records=len(verified),source_files=verified))
    destination=ROOT/'models/high_count_20261024';destination.mkdir(parents=True,exist_ok=True)
    target=destination/'01_conditional_high_count.onnx'
    if target.exists():assert audit.sha(target)==digest
    else:shutil.copy2(EXPORT/'model.onnx',target)
    metadata=dict(id='conditional_high_count_20261024',path=target.relative_to(ROOT).as_posix(),
        bytes=target.stat().st_size,sha256=digest,platform_score=None,status='exploratory_awaiting_platform',
        report='docs/HIGH_COUNT_20261024.md',base_sha256=audit.SOURCE_HASH,base_platform_score=.48,
        method='Frozen scored048;26-parameter positive conditional2/3/4 head; only lower base3/4 if high-head max probability>=.6; preserve base0/1/2.',
        training='600steps Adam lr.02, class/room balanced conditional binary losses + bias regularization; all19 true2/3/4 source records, seed42, no external/hidden labels.',
        limitations='Actual encoder saw all source labels; only two high-count rooms; independent proxy not improved. Source gains do not imply hidden gain.',
        local_validation=validation,resources=resources)
    audit.save(target.with_suffix('.json'),metadata)
    results=audit.read(WORK/'results.json');assert len(results)==16
    comparison=[]
    for origin in ('actual_diagnostic','independent_proxy'):
        for pair in (False,True):
            for seed in (42,2026):
                selected=[r for r in results if r['origin']==origin and r['pair_loss']==pair and r['seed']==seed]
                assert len(selected)==2
                comparison.append(dict(origin=origin,pair_loss=pair,seed=seed,
                    corrected=sum(r['metrics']['corrected'] for r in selected),broken=sum(r['metrics']['broken'] for r in selected),
                    by_room={r['room']:{k:r['metrics'][k] for k in ('corrected','broken')} for r in selected}))
    fixed=audit.read(WORK/'export/validation.json')
    audit.save(WORK/'summary.json',dict(candidate=metadata,comparison=comparison,fixed_consensus=fixed))
    audit.save(ROOT/'checks/high_count_20261024.json',dict(candidate=metadata,comparison=comparison,fixed_consensus=fixed))
    registry=audit.read(ROOT/'models/registry.json');assert registry['current_best']['platform_score']==.48
    registry['research_candidates']=[v for v in registry.get('research_candidates',[]) if v.get('id')!=metadata['id']]+[metadata]
    registry['next_submission']={k:metadata[k] for k in ('id','path','sha256','platform_score','status','report')}
    registry['latest_research']=dict(report=metadata['report'],status='conditional_high_count_candidate',best_scored=.48)
    registry['research_status']='conditional_high_count_awaiting_platform'
    audit.save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if not any(r['sha256']==digest for r in rows):
        rows.append(dict(model='26参数条件高人数读出',platform_score='',path=metadata['path'],sha256=digest,
            source='local research candidate; platform not evaluated',status='exploratory_awaiting_platform',
            submission_id='',availability='local',original_path=metadata['path'],archive_manifest=''))
        with p.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# 0.48平台持平后的高人数边界实验','',
        '用户反馈：上一份01_temporal_consensus80.onnx也为平台0.48。已记录其确切SHA，提交编号未提供。原始1人保护0.48保留为主对照，两份均保留；不能由相同显示分数推断隐藏逐样本预测相同。',
        '',f"下一份探索候选：`{metadata['path']}`，{metadata['bytes']/1e6:.2f}MB，平台尚未评测，SHA256 `{digest}`。",'',
        '## 为什么改查2/3/4','',
        '上一轮去偏置0→1在源数据纠正17/改坏5，平台仍0.48。本轮单独处理另一批错题，不叠加上一轮0→1规则，底座使用原始获分0.48。原始残余中2→3有40窗、2→4有14窗、3→4有41窗，高人数偏向过估。',
        '','固定对照：两路时间方向的去偏置argmax都为2或3，并且低于底座预测时才下调。原始窗口核对纠正26/改坏0，680/960（70.833%）；新增正确全为真实3人。它有时只是把真实2人的4误判改为3，仍是错题，不能算纠正。本轮固定规则仅保存在D盘作为对照，没有作为第二份提交文件混在活动模型目录。',
        '','## 新训练的条件读出','',
        '原网络已包含序数头，这次不是发明新的序数分类。我们冻结原四路线及四视角计算，读取每个视角的直接分类、原序数分布、静态专家分布，共12组现成证据；训练一个独立高人数读出，重新学习这些证据如何区分2/3/4。',
        '','两个条件问题：A=P(人数>2 | 人数≥2)，B=P(人数>3 | 人数>2)。最终高人数分布为P2=1−A、P3=A(1−B)、P4=AB。每个条件使用12个非负且和为1的证据权重及1个偏置，共26个新参数。输入是对应边界的有界对数比，没有直接使用房间名称或高维原始房间摘要。',
        '','训练用所有19条真实2/3/4记录，共380个原始窗口；原始0.48底座仍保留其全部数据训练所得能力。固定600步、Adam学习率0.02、类别/房间平衡损失及偏置正则。另比较同房间跨人数记录排序损失；未见稳定优势，导出普通条件损失版本。',
        '','只有底座预测3/4，且新读出选择更低人数、最高高人数概率≥0.6时采用新读出；原先预测0/1/2的输入输出原样保留。0.6是高人数三类条件概率门槛，不是经隐藏集验证的可靠度保证。',
        '','## 对照结果','',
        '2个外层留房间×2个新读出种子×有/无排序损失×实际/独立代理证据，共16次训练；另2次全数据拟合。真实高人数仅在a1/c1，因此每个独立外层折只剩一个高人数训练房间。训练侧还排除了原先固定的完整校准记录；阈值不在外层上搜索。',
        '','| 证据 | 排序损失 | 种子 | 纠正/改坏 | a1 | c1 |','|---|---|---:|---:|---|---|']
    for r in comparison:
        lines.append(f"| {r['origin']} | {'有' if r['pair_loss'] else '无'} | {r['seed']} | {r['corrected']}/{r['broken']} | {r['by_room']['a1']['corrected']}/{r['by_room']['a1']['broken']} | {r['by_room']['c1']['corrected']}/{r['by_room']['c1']['broken']} |")
    lines+=['','实际编码器见过全部源标签，所以这组外层结果仅反映新读出的记录/房间留出诊断，不是独立整管线测试。独立代理未产生收益；它的底座与实际获分模型差异很大，也不能据此保证平台不提升。保留此候选是用户要求的平台探索，不把本地结果当成成绩承诺。',
        '','## 实际ONNX与资源检查','',
        f"实际原始CSI核对全部{validation['raw_directly_checked']}个底座预测3/4窗口；其他{validation['unchanged_by_structure']}窗由原计算图与参数逐项一致及门控证明保留，没有冒称全960窗直接调用候选。最终正确{validation['correct']}/960（{validation['accuracy']:.3%}），纠正{validation['corrected']}、改坏{validation['broken']}。这是训练来源诊断。",'',
        '| 真实人数 | 原0.48正确 | 新候选正确 | 总数 |','|---|---:|---:|---:|']
    oldcm=[[215,3,2,0,0],[134,155,20,40,11],[0,0,66,40,14],[0,0,1,78,41],[0,0,0,0,140]]
    for k,row in enumerate(validation['confusion_matrix']):lines.append(f"| {k} | {oldcm[k][k]} | {row[k]} | {sum(row)} |")
    lines+=['','触发分支另由原0.48的中间证据＋独立NumPy条件公式验证，再检查最终单输出ONNX。已核对源文件SHA、输入输出形状/精度、嵌入权重、实际触发/不触发与零输入。没有重新计算额外主干，只复用现有四视角证据。',
        '',f"CPU4线程最慢样例均值{max(v['cpu_mean_s'] for v in resources['cases']):.4f}秒，文件{metadata['bytes']:,}字节，峰值提交内存{resources['peak_commit_bytes']/1e9:.3f}GB；本机<0.7秒、<100MB、<6GB检查通过。",'',
        '本地4人140窗没有损失，不代表隐藏4人不会被误下调；高人数只有两个源房间，泛化证据依旧不足。需平台反馈判断这轮是否有效，目标0.54仍未完成。',
        '','所有辅助权重、逐窗口与逐记录错题、完整对照和固定规则备选保存在 `D:/huawei2026-experiments/high_count_20261024/`。']
    (ROOT/'docs/HIGH_COUNT_20261024.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (destination/'README.md').write_text('下一份探索候选01_conditional_high_count.onnx。26参数高人数条件读出，平台尚未评测。详见docs/HIGH_COUNT_20261024.md。\n',encoding='utf-8')
    print(dict(path=str(target),validation=validation,bytes=metadata['bytes'],sha256=digest))


if __name__=='__main__':main()
