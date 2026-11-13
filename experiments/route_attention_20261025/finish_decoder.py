"""Publish only a numerically/resource checked eligible decoder candidate."""
import csv,shutil
from decoder_round import *
from support_round import ROUND

def main():
    assert sha(SOURCE)==SOURCE_SHA;rr=read(DEC/'results.json');sr=read(ROUND/'results.json');eligible=[v for v in rr if v['eligible']]
    report_path='docs/DECODER_CONSENSUS_20261025.md';reg=read(ROOT/'models/registry.json');meta=None;v=None;resource=None
    if eligible:
        s=read(DEC/'selection.json');folder=DEC/'export';v=read(folder/'validation.json');resource=read(folder/'resource_check.json');proof=read(folder/'structure_proof.json')
        assert v['raw_windows']==960 and resource['status']=='pass' and sha(folder/'model.onnx')==resource['sha256']==proof['candidate_sha256']
        assert v['correct']==s['rooms']['actual']['correct'] and v['base_correct']==679
        dest=ROOT/'models/decoder_consensus_20261025/01_dual_decoder_consensus.onnx';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(folder/'model.onnx',dest)
        meta=dict(id='dual_decoder_consensus_20261025',name='分类与有序人数头共识融合',path=dest.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=dest.stat().st_size,platform_score=None,status='awaiting_platform',source='checked local decoder experiment; platform unknown',report=report_path,experiment_root=str(DEC),base_sha256=SOURCE_SHA,base_platform_score=.54,selection=s,local_validation=v,resource_check=resource,training='No new neural fitting. Original054 and previously trained quarter-adapter all weights unchanged; no new bias.',method='Fixed current054 routing; 25% alternative categorical/ordinal probability pooling,75% original per-view pool. Scope'+s['scope'],caveats='Source previously trained; outer rooms repeatedly reused in development; not proven toreach060.')
        meta['weight_reuse']=read(folder/'weight_reuse.json');assert meta['weight_reuse']['candidate_sha256']==meta['sha256']
        save(dest.with_suffix('.json'),meta)
        for name in ('validation.json','resource_check.json','structure_proof.json','weight_reuse.json'):shutil.copy2(folder/name,dest.parent/name)
        reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('id')!=meta['id']]+[meta];reg['next_submission']=meta
        cp=ROOT/'models/platform_scores.csv'
        with cp.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        entry={k:'' for k in fields};entry.update(model=meta['name'],path=meta['path'],sha256=meta['sha256'],source=meta['source'],status=meta['status'],availability='retained',original_path=meta['path']);rows.append(entry)
        with cp.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    else:reg['next_submission']=None
    status='checked_decoder_candidate_awaiting_platform' if meta else 'support_and_decoder_completed_no_candidate'
    reg.update(research_status=status,latest_research=dict(report=report_path,status=status,best_scored=.54,experiment_root=str(DEC),new_training_runs=0,configurations=16,new_model_exported=bool(meta)),candidate_note='Both scored054 retained. Fixed label-free support gates and learned-head decoding compared; target060 remains unverified.')
    reg['target_status']=dict(target=.60,status='research_in_progress',best_known=.54);save(ROOT/'models/registry.json',reg)
    for path in (DEC,ROUND):
        snap=path/'source_snapshot';snap.mkdir(exist_ok=True)
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
    codes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')};save(ROOT/'checks/decoder_consensus_20261025.json',dict(base_sha256=SOURCE_SHA,code_sha256=codes,support_plan=read(ROUND/'plan.json'),decoder_plan=read(DEC/'plan.json'),support_results=sr,decoder_results=rr,published=meta))
    supports='\n'.join(f"|{x['family']}|{x['feature']}|{x['rooms']['actual']['correct']}|{289+x['proxy_net']}|{x['proxy_net']:+d}|否|" for x in sr)
    decoders='\n'.join(f"|{x['method']}|{x['scope']}|{x['rooms']['actual']['correct']}|{289+x['proxy_net']}|{x['proxy_net']:+d}|{'是' if x['eligible'] else '否'}|" for x in rr)
    if meta:
        s=meta['selection'];r=s['rooms'];table='\n'.join(f"|{k}|{r[k]['correct']-r[k]['corrected']+r[k]['broken']}/{r[k]['total']}|{r[k]['correct']}/{r[k]['total']}|{r[k]['corrected']}|{r[k]['broken']}|" for k in ('a1','c1','g1','e1'))
        class_table='\n'.join(f"|{k}|{v['baseline_confusion_matrix'][k][k]}|{v['confusion_matrix'][k][k]}|" for k in range(5))
        rng=np.random.default_rng(20261025);boot=np.zeros(10000);record_count=0
        for room in ('a1','c1','g1','e1'):
            pp=torch.load(DEC/f"{s['method']}_{s['scope']}_{room}.pt",weights_only=True);rows=torch.load(DEC/f'{room}.pt',weights_only=True)['rows'];ix=pp['test'];record_ids=sorted({rows[i]['identifier'] for i in ix});gains=[]
            yy=pp['labels'];g=(pp['logits'].argmax(-1)==yy).int()-(pp['base'].argmax(-1)==yy).int()
            for rid in record_ids:gains.append(sum(int(g[i]) for i in ix if rows[i]['identifier']==rid))
            gains=np.asarray(gains);boot+=gains[rng.integers(len(gains),size=(10000,len(gains)))].sum(1);record_count+=len(gains)
        interval=np.quantile(boot/880*100,[.025,.975]);save(DEC/'bootstrap.json',dict(unit='whole_record_stratified_by_outer_room',records=record_count,repeats=10000,seed=20261025,net_pp_interval95=interval.tolist(),purpose='descriptive uncertainty only, after development selection; not a blind-test confidence claim'))
        final=f'''最终候选`{meta['path']}`，{meta['bytes']/1e6:.2f}MB，SHA256 `{meta['sha256']}`。平台分数未知，最佳仍0.54，目标0.60未证实。

选择`{s['method']}`、`{s['scope']}`，来源实际ONNX {v['correct']}/960，对照679/960；纠正{v['corrected']}、改坏{v['broken']}。来源已用于训练，不能当独立测试。

|外层房间|基线|候选|纠正|改坏|
|---|---:|---:|---:|---:|
{table}

按原始记录整组、房间内分层重采样的描述性95%区间为{interval[0]:+.2f}到{interval[1]:+.2f}个百分点（{record_count}条记录，10000次）。该区间也受反复开发选择影响，不是独立盲测保证。即使本轮窗口不重叠，同一记录内的窗口仍相关，不能将880窗视为880个独立试验；本轮净增4窗只是有限的候选证据。

|来源真实人数|054正确|候选正确|
|---|---:|---:|
{class_table}

来源2人、4人都有退步，不能说每类改善；scope=all不保证054最终高人数窗口的分数逐值保持。保留的是旧路由条件和旧权重，不是所有旧预测。

实际960个原始窗口逐条核对提交ONNX及同图054控制，与独立Torch网络在相同实际预处理输出上概率最大误差{v['independent_encoder_probability_max_error']:.3g}。沿用原ONNX预处理节点逐字节核对；不声称独立原生预处理实现完全等价。候选所有16元素以上的参数/常量数组都逐字节出现在已评分054文件中（dtype/shape/content），没有新训练大参数，证据见weight_reuse.json。CPU4线程独立进程检查通过，样例平均最慢{max(c['cpu_mean_s'] for c in resource['cases']):.3f}秒/窗，峰值commit {resource['peak_commit_bytes']/1e9:.2f}GB，内嵌权重、接口和FLOAT输出核对通过。平台速度与成绩仍需实测。首次导出曾有匿名Cast节点重名，已在交付前修复；实际完整审计和资源检查针对修复后的最终SHA文件。
'''
    else:final='全部16种配置未通过固定条件，本轮没有导出新的模型。未执行ONNX导出和资源检查，不能声称产生更高分模型。建议保留两份实际已评分054，不为这些失败配置消耗提交次数。'
    report=f'''# 0.54后：支持度拒绝与双解码器融合（2026-10-25）

基线：`models/next053_20261025/01_g54_selective_lowcount.onnx`，用户平台0.54，SHA256 `{SOURCE_SHA}`。与队友Gprime054并列，两个文件保持原样。目标0.60尚未达到。

## 本轮问题与固定比较范围

上一轮80次训练显示小分支能提高来源正确率，但整房间开发代理退步。本轮先验证能否拒绝缺乏支持的修正，再拆开原模型中已经监督训练的两个人数头研究互补。**本轮没有新神经网络训练**，不能把16种推理配置称为16次训练。

仍使用48条比赛标注记录、960个600帧不重叠窗及4个留房间教师，实际ONNX提取的预处理缓存。来源诊断基线679/960，外层开发代理289/880。各外层教师与新增分支fit排除对应外层记录；代理房间反复用于历史和本轮开发，不是全新盲测，无法用这些结果宣称平台达到060。

## A：训练数据支持度＋双种子一致拒绝

保留上一轮pool/attention均衡残差与binaryMLP均衡分支，固定两个种子融合、强度0.5，不重新选择快照。支持特征分别为三路表示均值/标准差1536维与34维相对关系。先按fit整条记录取原型，在fit原型均值/标准差标准化空间求最近原型距离；阈值固定calibration记录窗距离的95%分位数，不使用其人数标签拟合阈值。没有扫分位数或强度。

只有距离落在支持范围且两个种子最终预测相同，才允许接管原低人数输出；其余退回054。全数据没有calibration，所以用fit留一条记录最近距离的95%分位数定阈值。实际来源推理原型包含本身记录，诊断偏乐观，已明确记录。

|分支|支持特征|来源正确/960|代理正确/880|代理净增|通过|
|---|---|---:|---:|---:|---|
{supports}

六种均失败。特征邻近与两个种子同意并不保证标签条件分布一致，不能将“熟悉”当“修正可靠”。这是本轮结果支持的限制，不证明所有拒绝机制都无效。

## B：分类头与有序人数头分开融合

原模型原本就有direct与ordinal两套输出，各占0.5。direct又合并共享分类器和四路专家；ordinal用是否有人以及CORN有序人数条件概率得到合法0–4分布。本轮没有新建或重训这些头。原权重、既有25%相位/时间适配权重、偏置、温度、最终1人+0.35全部保持。

每个原始/反向/接收排列视角分别保留75%原概率，加入25%候选池：direct偏重、ordinal偏重、熵置信度自适应（权重限0.25–0.75）、前两类margin自适应（同限制）、归一化几何均值。再比较允许全部窗口变化或只允许054最终0/1区域变化，共10种固定配置。

所有旧路由决定仍由原始未改分数计算：原视角1保护、时间对0/2保护、全6排列fallback、原054低人数是否切适配器。只改变该条路由选中位置的分数，避免把融合差异与保护规则改动混为一谈。新权重及置信度在每个视角、每个样本上计算，没有按房间特调。

原始与适配的direct/ordinal重组成log概率，分别与精确缓存逐项断言最大log误差<1e-5，通过全部5个教师/960窗/12视角。此检查是模型部分重组，不等同最终ONNX资源验证。

|融合|作用范围|来源正确/960|代理正确/880|代理净增|通过|
|---|---|---:|---:|---:|---|
{decoders}

选择条件在测试前保存：代理总净增、至少两个房间改善、任一房间下降不超过2pp、来源正确数≥679、来源4人正确至少135（基线136）。没有看结果再降低标准。若多个通过，按代理净增及最差房间变化选择，仍属于开发选择。

## 结果与提交状态

{final}

所有大缓存和实验留D盘`support_gate_20261025`及`decoder_consensus_20261025`，小型完整结果在`checks/decoder_consensus_20261025.json`，代码快照/哈希保留。没有Git推送，没有删除旧文件。当前平台最佳仍0.54，任何新候选都不得预填060。
'''
    (ROOT/report_path).write_text(report,encoding='utf-8')
    notice=f"本轮支持度拒绝/双解码器共16配置，{'已产生一个核对通过的待测候选，但平台未知' if meta else '均未通过固定筛选，未导出模型'}；已实测最佳054保留，目标060未达到。"
    for filename,link in [('README.md',report_path),('docs/SUMMARY.md','DECODER_CONSENSUS_20261025.md'),('models/README.md','../'+report_path)]:
        p=ROOT/filename;text=p.read_text(encoding='utf-8');heading,rest=text.split('\n',1);p.write_text(heading+'\n\n最新：'+notice+f'[实验记录]({link})。以下旧状态为历史。\n'+rest,encoding='utf-8')
    print('DEC_FINISHED',status,meta['path'] if meta else 'no_model',flush=True)
if __name__=='__main__':main()
