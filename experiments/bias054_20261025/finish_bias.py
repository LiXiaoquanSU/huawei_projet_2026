"""Publish one frozen054 final bias trial; record source/cross-room tradeoffs."""
import csv,shutil
from evaluate_bias import *

def main():
    s=read(BWORK/'selection.json');folder=BWORK/'export';v=read(folder/'validation.json');resource=read(folder/'resource_check.json');proof=read(folder/'structure_proof.json')
    assert s['eligible'] and v['raw_windows']==960 and resource['status']=='pass'
    assert sha(folder/'model.onnx')==resource['sha256']==proof['candidate_sha256'] and sha(BASES[s['base']]['path'])==BASES[s['base']]['sha256']
    assert v['correct']==s['rooms']['actual']['correct'] and v['base_correct']==s['source_base_correct']
    dest=ROOT/'models/bias054_20261025/01_g54_bias_2p020_0m010.onnx';dest.parent.mkdir(exist_ok=True,parents=True);assert not dest.exists();shutil.copy2(folder/'model.onnx',dest)
    first=read(BWORK/'results.json');follow=read(BWORK/'followup_results.json')
    bootstrap=np.zeros(10000);rng=np.random.default_rng(20261025);record_count=0
    for room in ('a1','c1','g1','e1'):
        z,y,rows=data(room,s['base']);new=z+z.new_tensor(s['delta']);ix=old.split(room,rows)[2];g=(new.argmax(-1)==y).int()-(z.argmax(-1)==y).int();ids=sorted({rows[i]['identifier'] for i in ix});record_count+=len(ids);gains=np.array([sum(int(g[i]) for i in ix if rows[i]['identifier']==rid) for rid in ids]);bootstrap+=gains[rng.integers(len(gains),size=(10000,len(gains)))].sum(1)
    ci=np.quantile(bootstrap/880*100,[.025,.975]);uncertainty=dict(records=record_count,repeats=10000,seed=20261025,proxy_net_pp=s['proxy_net']/880*100,stratified_record_bootstrap95_pp=ci.tolist(),scope='Descriptiveafteradaptive selection; notindependentblindtest significance.')
    save(BWORK/'uncertainty.json',uncertainty)
    meta=dict(id='g54_final_bias_20261025',name='原Gprime054最终偏置：空房-0.10，2人+0.20',path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),bytes=dest.stat().st_size,platform_score=None,status='awaiting_platform',owner='self_calibration_on_teammate054',source='new final-output bias trial; platform untested',report='docs/BIAS054_20261025.md',experiment_root=str(BWORK),base_sha256=proof['base_sha256'],base_platform_score=.54,delta=s['delta'],selection=s,local_validation=v,resource_check=resource,structure_proof=proof,uncertainty=uncertainty,training='No new backbone/headtraining. Fixedsmallbiastrials followedby4adaptive2Dcomparisons; existingweightsandinternalroutingbytepreserved.',caveats='Sourcepreviouslytrained; outerrooms repeatedlyuseddevelopment; adaptivecombinationselectedafterfirststageresults; emptyand4personsourceclasseslose; notprovenbetterthan054.')
    save(dest.with_suffix('.json'),meta)
    for name in ('validation.json','resource_check.json','structure_proof.json'):shutil.copy2(folder/name,dest.parent/name)
    reg=read(ROOT/'models/registry.json');assert reg['current_best']['platform_score']==.54 and reg['best_own']['platform_score']==.54
    reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('id')!=meta['id']]+[meta];reg['next_submission']=meta;reg['research_status']='g54_final_bias_checked_awaiting_platform';reg['latest_research']=dict(report=meta['report'],status='one_checked_bias_trial',experiment_root=str(BWORK),best_scored=.54,first_stage_nonzero_configs=48,adaptive_followups=4,new_training_runs=0)
    reg['candidate_note']='Decoderfusion user051 recorded. Returned to exact scoredGprime054; finalAdd[-.1,0,.2,0,0],crossroom net21/880 withmainc1gain18. Source+3butempty-3and4person-1. Notclaimedplatformimprovement.';save(ROOT/'models/registry.json',reg)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    entry={k:'' for k in fields};entry.update(model=meta['name'],path=meta['path'],sha256=meta['sha256'],source=meta['source'],status=meta['status'],availability='retained',original_path=meta['path']);rows.append(entry)
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    snapshot=BWORK/'source_snapshot';snapshot.mkdir(exist_ok=True);codes={}
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name);codes[p.name]=sha(p)
    save(ROOT/'checks/bias054_20261025.json',dict(meta=meta,plan=read(BWORK/'plan.json'),adaptive_plan=read(BWORK/'followup_plan.json'),first_stage=first,followup=follow,code_sha256=codes))
    keys={'class0_-0.10','class1_+0.20','class2_+0.10','class2_+0.20','class4_-0.10'}
    first_table='\n'.join(f"|{x['base']}|{x['name']}|{x['source_delta']:+d}|{x['proxy_net']:+d}|{x['rooms']['actual']['confusion_matrix'][4][4]}|{'是' if x['eligible'] else '否'}|" for x in first if x['name'] in keys)
    follow_table='\n'.join(f"|{x['base']}|{x['delta']}|{x['source_delta']:+d}|{x['proxy_net']:+d}|{(289 if x['base']=='selective' else 278)+x['proxy_net']}/880|{x['improving_rooms']}|{'是' if x['eligible'] else '否'}|" for x in follow)
    room_table='\n'.join(f"|{k}|{m['correct']-m['corrected']+m['broken']}/{m['total']}|{m['correct']}/{m['total']}|{m['corrected']}|{m['broken']}|" for k,m in s['rooms'].items() if k!='actual')
    classes='\n'.join(f"|{k}|{v['baseline_confusion_matrix'][k][k]}|{v['confusion_matrix'][k][k]}|" for k in range(5))
    report=f'''# 回到054的有限偏置试验（2026-10-25）

## 平台反馈

用户确认双解码器`01_dual_decoder_consensus.onnx`平台**0.51**，SHA256 `735a63b54294497923f3fe5a94dc5ed39bb03ee8b6f12c3ff2e93292ef84e63e`已登记，撤销其待测推荐。它的本地来源和代理小幅增加未能迁移到平台，不能再当作高于054的模型。两份已评分054保留原样。

新待测`{meta['path']}`，{meta['bytes']/1e6:.2f}MB，SHA256 `{meta['sha256']}`。**平台成绩未知；最好仍054，不声称达到060或保证小幅上涨。**

## 改了哪里

底座是队友原始Gprime054，SHA256 `{proof['base_sha256']}`；不是刚失败的051，也不是选择性更新054。只在其最终5维logit上加`[-0.10,0,+0.20,0,0]`，顺序为0/1/2/3/4人。

保持原内部五类偏置、最终1人+0.35、原视角1保护、时间对0/2保护、6Rx排列×时间正反fallback以及全部权重与预处理节点。新图仅追加1个Add与5个FLOAT常数。它改变最终类别边界，不增加特征信息。空房相对分数乘约0.905、2人乘约1.221，是分数比变化，不是准确率或概率直接加20个百分点。原路由规则不改，但最终预测仍可能改变。

原文件末尾对1人额外加0.35；新2人+0.20使这两个类之间的末尾附加优势从1人领先0.35缩到领先0.15，保留1人相对3/4人的附加优势。同时空房-0.10使末尾1人相对0人的附加优势增加到0.45。这只是在解释末尾常数的相对差，不是整个内部bias向量或对隐藏人数先验的估计。与曾经统一加大1人偏置的实验不同，但仍可能增加空房误报。

## 比较范围和防止夸大

第一阶段提前保存：每类单独±0.10/±0.20，共20项；人数线性倾斜±0.05两项；0/1对比±0.10两项。对选择性054、原Gprime054各24项非零偏置，加各自基线，共50个结果。**相同数值的偏置用于各外层教师**，没有为每个房间单独找最优值。

筛选为代理总净增、至少两房间提高、任一房间损失≤2pp、来源总数不减、来源4人至多损1。第一阶段没有通过者。以下是关键方向，完整50项在D盘与checks记录。

|底座|偏置|来源净增/960|代理净增/880|来源4人正确|通过|
|---|---|---:|---:|---:|---|
{first_table}

2人+0.20代理收益大，但来源净损1；空房-0.10来源增加、代理仅一个房间增加，均不满足全部条件。随后另存followup_plan.json，只比较这两个方向的4个组合：2人+0.10或+0.20，与空房-0.10，两份底座各2项。**这是看到首轮结果后提出的自适应开发，并非独立验证，也不能称整轮预注册。没有为了选中候选放宽原标准。**

|底座|最终残差|来源净增|代理净增|代理正确数|改善房间数|通过|
|---|---|---:|---:|---:|---:|---|
{follow_table}

按预设选择代理总净增最多者，然后依次比较最差房间、改判数量和幅度，因而本轮回到原Gprime文件。**Gprime仅在相对净增上更大；选择性版本同残差的绝对代理正确数306/880反而高于Gprime299/880，因为原基线也更高，不能据净增就宣称Gprime总体泛化更强。** 两个底座平台都已评分054，本轮仅先送一份明确的单变量偏置对照，未导出其余组合。第一阶段单类4人降低等方案虽来源上升，却损失多例4人正确，淘汰；不沿着来源最高值继续扩大偏置。

## 留房间开发结果

{278+s['proxy_net']}/880，对照278/880，净增{s['proxy_net']}（+{s['proxy_net']/880*100:.2f}个百分点）。注意之前选择性054的基线为289/880，**不是本轮Gprime的278**，不能混用。

|外层房间|原Gprime054|新偏置|纠正|改坏|
|---|---:|---:|---:|---:|
{room_table}

主要收益集中c1（+18），a1仅+1、g1仅+2、e1持平，不能当作广泛且显著的跨房间改善。训练/校准/外层测试按原始记录隔离，教师fit排除外层房间；但这些房间反复用于历史及本轮偏置选择，仍只是开发代理，不能预测隐藏平台涨分。

整条记录、房间内分层bootstrap的描述性95%净增区间为{ci[0]:+.2f}至{ci[1]:+.2f}个百分点（44条记录、10000次、固定随机种子）。该区间不校正自适应挑选，不能据此宣称独立显著性，也不能把同一记录20窗视为20个独立试验。

## 实际960原始窗口核对

新候选**{v['correct']}/960（{v['accuracy']:.2%}）**，原Gprime{v['base_correct']}/960。纠正{v['corrected']}、改坏{v['broken']}，净增{v['correct']-v['base_correct']}。来源数据用于过训练，只作诊断。

|真实人数|原正确|新正确|
|---|---:|---:|
{classes}

来源空房损3、4人损1，1人增7，2/3人正确数不变。**2人偏置并不意味着来源2人识别已提高**；它在来源还引入了4→2误判。偏置有取舍，不能保证隐藏数据同样收益。已经连续出现“本地略升、平台下降”，所以仍需一次明确的单变量平台对照。

原图逐节点、全部初始化参数和接口逐字节核对，只允许原最终输出重命名接到新增Add。实际960原始窗口逐条运行，新输出与同图原输出+残差逐值相等；旧精确特征缓存与实际原图概率最大误差{v['cached_network_probability_max_error']:.3g}，边界argmax差异{v['borderline_argmax_disagreements']}。没有依靠近似原生预处理代替原图。

CPU4线程独立资源检查通过，各样例平均最慢{max(c['cpu_mean_s'] for c in resource['cases']):.3f}秒/窗，95分位最慢{max(c['cpu_p95_s'] for c in resource['cases']):.3f}秒，峰值commit {resource['peak_commit_bytes']/1e9:.2f}GB。检查条件是均值<0.7秒，不能据此声称每一窗都小于0.7秒；本机时间余量有限，但相对已评分原图只多一次5元素加法，没有额外主干推理。权重内嵌，接口frames[600,1,3,1024,2]、intervals[600]→5类FLOAT不变。平台机器速度和分数仍由实测确认。

大记录、逐条预测和程序快照留D:/huawei2026-experiments/bias054_20261025；C盘仅新增最终一个偏置候选、小型记录和报告。0.51与两份054文件保留，未推送Git。
'''
    (ROOT/meta['report']).write_text(report,encoding='utf-8')
    for filename,link in [('README.md',meta['report']),('docs/SUMMARY.md','BIAS054_20261025.md'),('models/README.md','../'+meta['report'])]:
        p=ROOT/filename;text=p.read_text(encoding='utf-8');heading,rest=text.split('\n',1);p.write_text(heading+'\n\n最新：双头融合平台051，最佳仍两份054。新待测为原Gprime054最终偏置[-.10,0,+.20,0,0]，无新训练；代理净增21但集中c1，来源空房/4人有损失，不保证平台提高。'+f'[本轮记录]({link})。以下旧状态为历史。\n'+rest,encoding='utf-8')
    print('BIAS_PUBLISHED',dest,meta['sha256'],flush=True)
if __name__=='__main__':main()
