"""Publish exactly one checked exploratory artifact; keep scored .49 best."""
import csv,shutil
from run import *

def main():
    export=WORK/'exports/calibrated_consistent_s0.25_mix1.0'
    resources=read(export/'resource_check.json');validation=read(export/'validation.json');proof=read(export/'structure_proof.json')
    assert resources['status']=='pass' and resources['sha256']==proof['candidate_sha256']==sha(export/'model.onnx')
    assert validation['raw_windows']==960 and validation['zero_adapter_actual_raw_record_identity_checks']==48
    registry=read(ROOT/'models/registry.json');best=registry['current_best']
    assert best['platform_score']==.49 and sha(ROOT/best['path'])==best['sha256']
    destination=ROOT/'models/temporal_adapter_20261024/01_temporal_adapter_consistent.onnx'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.with_suffix('.json').exists():assert read(destination.with_suffix('.json'))['platform_score'] is None,'Never overwrite a scored candidate'
    shutil.copy2(export/'model.onnx',destination);assert sha(destination)==resources['sha256']
    proxies=read(WORK/'calibrated_proxy_results.json')
    px=[]
    for r in proxies:
        if r['method']!='calibrated_consistent':continue
        v=next(v for v in r['variants'] if v['scale']==.25 and v['mix']==1.)
        px.append(dict(room=r['room'],chosen_step=r['chosen_step'],**v['test']))
    conditional=read(WORK/'record_holdout_results.json')
    meta=dict(id='temporal_adapter_consistent_20261024',name='固定0.49校正的时间残差一致性适配器',
      path=destination.relative_to(ROOT).as_posix(),sha256=resources['sha256'],bytes=destination.stat().st_size,
      platform_score=None,status='awaiting_platform',base_sha256=best['sha256'],base_platform_score=.49,
      owner='self_adapter_on_scored_backbone',report='docs/TEMPORAL_ADAPTER_20261024.md',
      method='Only16672 temporal adapter parameters trained; residualscale0.25. Same original4views/fusion/guards and fixed044bias/final+0.35; no external or hidden labels.',
      checkpoint=str(WORK/'calibrated_consistent_0.25.pt'),experiment_root=str(WORK),
      local_validation=validation,resources=resources,independent_proxy_room_results=px,
      independent_proxy_net=sum(r['corrected']-r['broken'] for r in px),
      conditional_record_holdout=conditional,
      caveats='Full source trained backbone saw all source labels. Record holdout excludes adapter fitting only. Room proxy net+2/880 is weak and g1 degrades17; exploratory submission, not demonstrated hidden improvement.',
      submission_id=None)
    save(destination.with_suffix('.json'),meta)
    for name in ('structure_proof.json','resource_check.json','validation.json'):
        shutil.copy2(export/name,destination.parent/name)
    registry['next_submission']=meta
    registry['research_candidates']=[v for v in registry.get('research_candidates',[]) if v.get('id')!=meta['id']]+[meta]
    registry['research_status']='temporal_adapter_checked_awaiting_platform'
    registry['candidate_note']='Scored best remains0.49. One small residual adapter candidate checked; no hidden accuracy claim. Experiments on D.'
    registry['latest_research']=dict(report=meta['report'],status='38_adapter_fits_one_checked_export',best_scored=.49)
    save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['sha256']!=meta['sha256']]
    new={k:'' for k in fields};new.update(model=meta['name'],path=meta['path'],sha256=meta['sha256'],source='new exploratory adapter; platform not evaluated',status='awaiting_platform',availability='retained',original_path=meta['path'])
    rows.append(new)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    save(ROOT/'checks/temporal_adapter_20261024.json',dict(metadata=meta,proof=proof,code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    cm=validation['confusion_matrix'];correct=validation['correct'];total=validation['total']
    roomtable='\n'.join(f"|{r['room']}|{r['corrected']}|{r['broken']}|{r['corrected']-r['broken']:+d}|" for r in px)
    classtable='\n'.join(f"|{k}|{old}|{cm[k][k]}|{cm[k][k]-old:+d}|" for k,old in enumerate([206,198,66,78,140]))
    text=f'''# 时间残差适配器实验（2026-10-24）

## 当前结论与提交文件

已评分最佳仍为0.49。下一份探索性候选：`{meta['path']}`；平台尚未评测，不能宣称已超过0.49。

- SHA256：`{meta['sha256']}`。
- 文件：{meta['bytes']/1e6:.2f} MB，单文件内嵌全部权重，无外置数据。
- 训练、缓存及其余实验参数全部在`{WORK.as_posix()}`；C盘只保留这一个新提交模型与精简记录。原0.49字节不变。

## 结构与训练

原静态、动态、相位、时间四路线不变。只在256维时间编码后加入256→32→256、SiLU残差适配器，新增可训练参数16,672。上投影零初始化，初始化为恒等映射；正式使用残差更新的0.25倍，以控制表示漂移。

适配后的时间证据进入原专家、门控、共享分类器、presence及人数序数输出。保持原4个时间/Rx视角、原融合和0/2保护、原视角1人保护、原0.44偏置和最终1人+0.35。各规则作用于更新后的证据，触发集合会变化；并非所有原先0/1预测都被结构性保护。

原编码器及读出权重全部冻结且eval；没有把Dropout/MixStyle打开。只用比赛原有48条标注记录、960个600帧窗口及其正确时间/Rx变换；未加入外部数据、未知标签或伪造动作标签。

AdamW，lr0.001、weight_decay0.01，seed2026，全批次75步，梯度范数截断1。损失为0.75最终校正融合分数的监督交叉熵＋0.25逐视角校正分数交叉熵（标签平滑0.02），加0.2原未校正教师KL、0.2视角JS一致性、0.01残差均方。按类别、同类房间平衡；不重新学习任何类别偏置。全数据75步由各代理内部整记录校准所选步数的中位数确定。

## 完成的对照

7种训练方式×4个外层留房间代理＋7次全数据拟合＋3次整记录适配器留出＝38次优化训练。25/75/150步为同一训练的快照，0.25/0.5/1残差比例和0.5/1概率混合为推理消融，没有算成额外训练。

1. 未校正输出监督：原始视角CE训练会改善部分2/3人，但部署校正后1人召回明显变差，不推荐。
2. 原始监督＋一致性、再加同类跨房间约束：没有解决上述部署目标错位。
3. 改用固定校正后的实际输出监督：得到局部收益；强适配的跨房间退步明显。
4. 实际输出监督＋一致性：选小残差0.25，不额外做概率混合。
5. 再加同人数0/1跨房间中心对齐或困难类别/房间组的平滑最大损失：没有稳定超过第4组，不推荐额外复杂度。

困难组版全强度源数据可达778/960，但独立房间代理明显退步，故没有选择本地最高分。该结论再次说明训练来源的总正确率不足以预测平台。

## 实际ONNX原始输入核查

48条记录的960个原始CSI/ATI窗口全部运行新ONNX，正确{correct}/{total}（{correct/total:.3%}），相对实际0.49对照纠正{validation['corrected']}、改坏{validation['broken']}，净增{validation['corrected']-validation['broken']}。

|真实人数|原0.49正确数|本候选正确数|变化|
|---|---:|---:|---:|
{classtable}

以上是训练来源诊断，不是独立测试准确率。完整逐窗输出在D盘`exports/calibrated_consistent_s0.25_mix1.0/all_predictions.json`。

新增头在原ONNX中间证据上与独立PyTorch读出逐窗比对，最大logit差{validation['independent_torch_readout_logit_max_error']:.3g}；原0.49对照与已核查缓存最大logit差{validation['frozen_actual049_baseline_logit_max_error']:.3g}。另对每条记录首窗验证零适配器恒等，共48次。原9065节点和全部原初始化权重字节保持，只接入507个新读出节点，没有增加编码器推理次数。

## 泛化证据与限制

整房间排除代理的编码器和新适配器都没有见过外层房间标签；适配器仅使用该编码器原fit记录，25/75/150步由内部calibration整记录NLL选择。保留每个代理自己既有的温度/偏置，不将实际0.44偏置硬套给它。

|外层房间|纠正|改坏|净增|
|---|---:|---:|---:|
{roomtable}

四个代理共880窗，净增2，仅约0.23个百分点，g1退步17窗。证据较弱，且代理历史反复用于研究、只覆盖少量房间；不能把这当作可靠的隐藏集提升。尤其全房间留出同时留出了layout，与老师实际评测未必相同。

三折整记录适配器留出分别净增7、11、12，合计纠正33/改坏3、718/960。但历史冻结主干已见这些记录标签，只有适配器本轮未见；属于条件增益检查，不能称端到端独立交叉验证。

## 运行核查与后续

CPU隔离进程检查通过：输入frames float32[600,1,3,1024,2]、intervals float32[600]，输出logits float32[5]；4线程最慢用例平均{max(r['cpu_mean_s'] for r in resources['cases']):.3f}秒，峰值提交内存{resources['peak_commit_bytes']/1e9:.2f}GB；小于本地检查的100MB、0.7秒、6GB约束。不能保证服务器下载、网络或硬件条件。

建议只提交这一个温和候选来验证表示更新能否迁移；保留0.49作精确控制。若仍无提升，不继续加大适配强度或堆偏置，下一步需针对跨房间失败的表示/采样假设重新检验。平台成绩由用户后续反馈记录。
'''
    (ROOT/meta['report']).write_text(text,encoding='utf-8')
    (ROOT/'models/README.md').write_text(f'''# 模型登记

已评分最高为0.49：`final_bias_20261024/01_final_bias_one035.onnx`。以registry.json的current_best为准，保持原字节。

下一份待提交：`temporal_adapter_20261024/01_temporal_adapter_consistent.onnx`，{meta['bytes']/1e6:.2f}MB，时间残差适配器＋一致性训练，平台分数未知。核查与训练限制见[实验记录](../docs/TEMPORAL_ADAPTER_20261024.md)。

0.47的4/12视角版和其他获分里程碑仍保留对照。platform_scores.csv保留完整历史；空白分数表示未评测，不是零分。availability=archived时由original_path及归档清单恢复。

本轮只在本机保存新候选，没有推送GitHub。训练记录与其他实验参数在D盘。
''',encoding='utf-8')
    for filename,old in [('README.md','**最新平台反馈：**'),('docs/SUMMARY.md','最新反馈：')]:
        p=ROOT/filename;content=p.read_text(encoding='utf-8')
        intro=f"本轮完成38次适配器训练，已导出并核查一个候选`{meta['path']}`（{meta['bytes']/1e6:.2f}MB）。原始960窗{correct}/{total}，纠正{validation['corrected']}/改坏{validation['broken']}；这是训练来源诊断。整房间代理净增仅2/880且g1退步，平台尚未测，最佳仍0.49。[本轮实验]({'docs/' if filename=='README.md' else ''}TEMPORAL_ADAPTER_20261024.md)。\n\n"
        if filename=='README.md':
            content=content.replace('**最新平台反馈：** 总偏置0.50版获分0.48，低于0.35版的0.49，停止继续增大该偏置。目前没有新候选等待提交。已完成冻结四路线的源记录级表示诊断，下一轮优先验证原权重时间适配器与受控一致性训练。[复盘与方案](docs/AFTER_BIAS050_048.md)。',intro.strip())
        else:content=content.replace('# 经验总结\n\n','# 经验总结\n\n'+intro,1)
        p.write_text(content,encoding='utf-8')
    print('PUBLISHED',destination,'best remains0.49',flush=True)

if __name__=='__main__':main()
