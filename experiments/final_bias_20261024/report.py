"""Publish a bounded final-bias exploration, preserving all scored artifacts."""
import csv
import shutil
from pathlib import Path
from build import ROOT,WORK,EXPORT,SOURCE,SOURCE_HASH,read,save,sha


def main():
    destination=ROOT/'models/final_bias_20261024';target=destination/'01_final_bias_one035.onnx'
    if target.with_suffix('.json').exists() and read(target.with_suffix('.json')).get('platform_score') is not None:
        print('Scored artifact already published; preserve later feedback.');return
    check=read(EXPORT/'resource_check.json');validation=read(EXPORT/'validation.json');summary=read(WORK/'summary.json')
    digest=sha(EXPORT/'model.onnx');assert sha(SOURCE)==SOURCE_HASH
    assert check['status']=='pass' and check['sha256']==validation['candidate_sha256']==digest
    assert validation['all_changed_windows_directly_checked']==53 and validation['output_logit_max_error']<1e-5
    destination.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==digest
    else:shutil.copy2(EXPORT/'model.onnx',target)
    metadata=dict(id='final_bias_one035_20261024',path=target.relative_to(ROOT).as_posix(),sha256=digest,
        bytes=target.stat().st_size,platform_score=None,status='exploratory_awaiting_platform',
        base_sha256=SOURCE_HASH,base_platform_score=.48,delta=summary['chosen_delta'],
        report='docs/FINAL_BIAS_20261024.md',method='Frozen original048 logits + [0,.35,0,0,0]; one final Add. No re-evaluation of original routing.',
        rationale='Explore posterior class tradeoff after source-only high-count specialist dropped platform048->045. No hidden prior assumption.',
        caveats='Calibration subsets overlap; frozen actual encoder saw source labels. Chosen fixed .35 proxy gains mixed across rooms. Not predicted to beat .48.',
        local_validation=validation,resources=check)
    save(target.with_suffix('.json'),metadata)
    folds=read(WORK/'fold_results.json');assert len(folds)==24
    comparisons=[]
    for origin in ('actual_diagnostic','independent_proxy'):
        for method in ('one_positive','all_regularized','one_fixed035'):
            selected=[r for r in folds if r['origin']==origin and r['method']==method]
            assert len(selected)==4
            comparisons.append(dict(origin=origin,method=method,
                corrected=sum(r['metrics']['corrected'] for r in selected),broken=sum(r['metrics']['broken'] for r in selected),
                by_room={r['room']:dict(delta=r['delta'],corrected=r['metrics']['corrected'],broken=r['metrics']['broken']) for r in selected}))
    report=dict(candidate=metadata,comparisons=comparisons,source_diagnostics=summary)
    save(WORK/'report.json',report);save(ROOT/'checks/final_bias_20261024.json',report)
    registry=read(ROOT/'models/registry.json');assert registry['current_best']['platform_score']==.48
    registry['research_candidates']=[v for v in registry.get('research_candidates',[]) if v.get('id')!=metadata['id']]+[metadata]
    registry['next_submission']={k:metadata[k] for k in ('id','path','sha256','platform_score','status','report')}
    registry['latest_research']=dict(report=metadata['report'],status='bounded_final_bias_exploration',best_scored=.48)
    registry['research_status']='final_bias_one035_awaiting_platform'
    save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if not any(r['sha256']==digest for r in rows):
        rows.append(dict(model='原048最终1人偏置加0.35',platform_score='',path=metadata['path'],sha256=digest,
            source='local bias exploration; platform not evaluated',status='exploratory_awaiting_platform',submission_id='',
            availability='local',original_path=metadata['path'],archive_manifest=''))
        with p.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# 高人数0.45复盘与最终偏置实验','',
        '**已记录用户反馈：01_conditional_high_count.onnx平台0.45，低于两份0.48；没有覆盖其确切文件与哈希。** 提交编号未知，原始0.48继续作为主对照。',
        '',f"下一份探索候选：`{metadata['path']}`，{metadata['bytes']/1e6:.2f}MB，SHA256 `{digest}`，平台尚未评测。",'',
        '## 0.45带来的经验','',
        '上一轮688/960、本地纠正34/改坏0，平台却从0.48降到0.45。本地高人数只在a1/c1，且原编码器已看过全部源标签；小读出只有26个参数仍可能学到不迁移的边界。独立代理没有收益，是应保留的负面证据。不能再把训练来源正确率更高作为平台选型的主要理由。',
        '','在平台使用相同测试集和一致评测的前提下，上一轮只会下调原始底座3/4的部分判断，因此总体损失来自这些被改动的输入；不能从总体0.45判断具体是3人还是4人退步，也不能推出隐藏类别比例。ONNX数值/资源检查通过只能排除已检验的实现问题，不能证明根因唯一是房间过拟合。',
        '','下一步先保留已有效的高人数相对分数，检验输出校准；后续需要真实独立房间的表示或变换一致性改进。更多重叠窗口和更复杂头不能替代新房间。',
        '','## 同权重改偏置能不能提高','',
        '可以有机会：已提供的0.42/0.44文件逐项对比只有最后偏置不同，平台确有提升；后续0.47/0.48还依赖输入变换和保护规则。偏置移动类别边界，不生成新信号特征，可能救回边界错误，也会损害另一类。不是偏置调大就持续涨分。',
        '','在类别条件分布不变、后验估计可用等假设下，logit先验校正可以写为加log目标先验−log有效源先验；本题换房间可能连P(CSI|人数)都改变。训练又使用类别/房间平衡采样，原始48条记录的人数比例不等于模型有效先验。不能未经验证就当作纯先验偏移，或强迫老师类别均匀。方法依据：[Logit adjustment](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)、[Label shift条件](https://proceedings.mlr.press/v80/lipton18a.html)。',
        '','## 本轮方法和选择','',
        '固定原始获分0.48，不用0.45高人数头，不叠加另一份temporal80规则。对最终五维输出加常数，原内部偏置、时间变换与保护路由逐节点保留。这样能把最终偏置的作用与路由变化分开。',
        '','对照：1人固定加0.35/0.5、原偏置收缩10%、4人加0.2诊断；训练侧完整校准记录拟合单类非负残差或五类有界残差。固定500步投影梯度、步长0.1、L2系数0.2；单类范围[0,0.35]，五类各[-0.35,0.35]。本轮没有重新训练主干。',
        '','| 源数据诊断 | 纠正 | 改坏 | 正确/960 |','|---|---:|---:|---:|']
    for name,v in summary['candidates'].items():
        if name=='one_calibration_median':continue
        m=v['metrics'];lines.append(f"| {name} | {m['corrected']} | {m['broken']} | {m['correct']} |")
    lines+=['','选择单类+0.35：提高1人分数，不直接改变0/2/3/4之间的相对顺序；1人仍可能超过这些类，绝非保证高人数永不受影响。0.5虽然源数据更高，新增误报也更多，因此没有继续扩大范围或按本地最高值选型。五类共同校准会再次下压4人，不作为这次提交变量。',
        '','四个留房间方案对应的训练侧校准集拟合1人残差约0.227、0.350、0.350、0.350，中位数0.350。其中校准子集重叠，g1/e1两组相同，不能当四次独立交叉拟合证据。实际编码器见过外层标签；本轮固定数值也已在开发数据上检查，不是未见数据上的预注册最优偏置。',
        '','## 留房间与独立代理的风险','',
        '| 来源 | 方法 | 纠正/改坏（880窗） | a1 | c1 | g1 | e1 |','|---|---|---|---|---|---|']
    for r in comparisons:
        per=[f"{r['by_room'][room]['corrected']}/{r['by_room'][room]['broken']}" for room in ('a1','c1','g1','e1')]
        lines.append(f"| {r['origin']} | {r['method']} | {r['corrected']}/{r['broken']} | {' | '.join(per)} |")
    lines+=['','独立代理自己拟合的单类幅度为0、0.093、0.0039、0.0102，比实际底座小；把同一个+0.35直接移过去时，收益偏向g1、c1反而退步。不能把“代理各自小偏置有收益”偷换成“+0.35已稳定验证”。保留这个候选是一次单变量平台探索，不是已证明泛化更好。',
        '','## 导出验证与代价','',
        f"新图只增加1个Add和5个float32残差。全部960窗结果由此前实际ONNX核对过的0.48输出加float32残差推导；另直接核对{validation['directly_raw_checked']}个原始输入，包括全部{validation['all_changed_windows_directly_checked']}个改判窗口和边界样例，不冒称960次新文件直接推理。原输出缓存差和候选参考差均为0。",'',
        f"最终688/960（{validation['accuracy']:.3%}），纠正43、改坏9。1人正确155→198，空房215→206，2/3/4人正确数66/78/140未变。与上一轮0.45模型本地总正确数相同，但改对/改错的样本、人数类别和机制不同，不能据相同本地数字预测平台。",'',
        f"文件{metadata['bytes']:,}字节，CPU4线程最慢样例均值{max(v['cpu_mean_s'] for v in check['cases']):.4f}秒，峰值提交内存{check['peak_commit_bytes']/1e9:.3f}GB；接口、嵌入权重、独立原0.48+NumPy参考、<100MB/<0.7秒/<6GB检查通过。",'',
        '这次最直接的代价是9个新增空房误报。目标仍是平台0.54，当前最好仍0.48，候选没有预测平台分数。',
        '','## 后续提升空间','',
        '1. 若偏置有效，围绕同一底座做少量受限单变量对照，观察平台是否支持这个方向，避免凭多次提交盲目拟合测试机。',
        '2. 若偏置也无益，瓶颈更可能不只是统一类别阈值；应在原获分权重上验证少量残差适配/变换一致性，并保留不变底座对照，而不是再从头放大网络。',
        '3. U数据可作域变化诊断或自监督表示学习，但不能因为无标签就套用类别均匀或把教师伪标签当真值；先验证方法假设。',
        '','所有实验与逐记录错题：`D:/huawei2026-experiments/final_bias_20261024/`。']
    (ROOT/'docs/FINAL_BIAS_20261024.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (destination/'README.md').write_text('未评测探索候选01_final_bias_one035.onnx：原始0.48最终1人分数加0.35，内部规则不变。详见docs/FINAL_BIAS_20261024.md。\n',encoding='utf-8')
    print(dict(path=str(target),bytes=metadata['bytes'],sha256=digest,platform_score=None,local_correct=688,corrected=43,broken=9))


if __name__=='__main__':main()
