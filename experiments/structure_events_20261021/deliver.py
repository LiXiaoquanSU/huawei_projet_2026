"""Export, independently check, audit and summarize the selected new structure."""
import csv,json,shutil,subprocess,sys
from pathlib import Path
import blosc2
import numpy as np
import torch
from train import ROOT,WORK,read,save,sha,load,seed,build

class Submission(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames,intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_features(t.float(),s.float())[0]

def main():
    if (WORK/'platform_feedback_038.json').exists():raise RuntimeError('This model is already evaluated at0.38; use a new experiment directory instead of overwriting feedback.')
    torch.set_num_threads(4);selection=read(WORK/'selection.json');v,s=selection['architecture'],selection['seed'];folder=WORK/'fits'/f'{v}_s{s}_full'
    cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=True);model=build(v).eval();model.load_state_dict(cp['model_state']);wrapper=Submission(model).eval()
    dest=WORK/'export';dest.mkdir(exist_ok=True);references=[]
    for path in [ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero']:
        if path=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(path) as d:x,dt=[torch.from_numpy(d[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():z=wrapper(x,dt)
        assert z.shape==(5,) and torch.isfinite(z).all();references.append(dict(input=str(path),logits=z.tolist()))
        if len(references)==1:example=x,dt
    save(dest/'references.json',references)
    torch.onnx.export(wrapper,example,str(dest/'model.onnx'),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(dest)],check=True)
    resource=read(dest/'resource_check.json');target=ROOT/'models/structure_events_20261021'/f'01_{v}_seed{s}.onnx';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==resource['sha256']
    else:shutil.copy2(dest/'model.onnx',target)
    alias=ROOT/'submissions/structure_events_20261021';alias.mkdir(parents=True,exist_ok=True)
    if (alias/'model.onnx').exists():assert sha(alias/'model.onnx')==resource['sha256']
    else:shutil.copy2(target,alias/'model.onnx')
    meta=dict(architecture=v,seed=s,path=target.relative_to(ROOT).as_posix(),alias='submissions/structure_events_20261021/model.onnx',bytes=resource['bytes'],sha256=resource['sha256'],parameters=cp['config']['parameters'],platform_score=None,status='new_architecture_platform_unknown',config=cp['config'],resource=resource,checkpoint=str(folder/'final.pt'))
    meta.update(stress_gate_passed=selection.get('stress_gate_passed'),recommendation='research_only_not_replacement_for_platform037')
    save(target.with_suffix('.json'),meta);save(alias/'metadata.json',meta);save(WORK/'primary_candidate.json',meta)
    # Check records, source cache identities and learned structural parameters.
    d=load();assert torch.isfinite(d['t']).all() and torch.isfinite(d['s']).all();fits=[]
    r=d['rows'][0];raw=Path('D:/huawei2026-data/labeled');a,b=int(r['start']),int(r['stop'])
    c=blosc2.load_array(str(raw/f"CSI_{r['identifier']}.b2"));ati=blosc2.load_array(str(raw/f"ATI_{r['identifier']}.b2"))
    x=torch.from_numpy(c[a:b].view(np.float32).reshape(600,1,3,1024,2).copy());dt=torch.from_numpy(ati[a:b].copy());dt[-1]=0
    with torch.no_grad():
        t,sm=model.preprocess(x[None].double(),dt[None].double());t,sm=t.float(),sm.float()
        feature_error=float((t-d['t'][:1]).abs().max());prediction_error=float((model.forward_features(t,sm)-model.forward_features(d['t'][:1],d['s'][:1])).abs().max())
    assert feature_error<.001 and prediction_error<.005,(feature_error,prediction_error)
    for p in sorted((WORK/'fits').glob('*/metrics.json')):
        m=read(p)
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        ids={r['identifier'] for r in tr}
        if m['config']['fold']=='full':assert len(ids)==m['actual_records']==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            assert ids.isdisjoint({r['identifier'] for r in va}) and all(r['room'] not in ('unknown',m['config']['fold']) for r in tr)
        assert not m['config']['external_data'];fits.append(m)
    assert len(fits) in (28,29),len(fits)
    seed(s);initial=build(v).state_dict();changes={}
    prefixes=(['temporal.'] if v in ('events','combined') else [])+(['relations.','relation_gate'] if v in ('relations','combined') else [])
    if v=='event_residual':prefixes=['temporal.events.','temporal.gate']
    for prefix in prefixes:
        delta=sum(float((value-initial[k]).square().sum()) for k,value in cp['model_state'].items() if k.startswith(prefix));assert delta>0;changes[prefix]=delta
    if hasattr(model,'relations'):
        with torch.no_grad():maps=model.relations.matrices(d['t'][:2]);zero=model.relations.matrices(torch.zeros_like(d['t'][:2]))
        assert maps.shape==(2,27,32,32) and torch.isfinite(maps).all() and zero.abs().max()==0
    old=read(ROOT/'checks/restarts_cleanup_20261020.json')['model_hashes']
    for name,expected in old.items():assert sha(ROOT/name)==expected
    prior=read(ROOT/'models/registry.json');assert prior['data_revision_20261021']['platform_score']==.35 and prior['structured_v1']['score']==.37
    assert sha(ROOT/'submissions/data_revision_20261021/model.onnx')=='21e7a4d38b650133e8ba9c3c0e335c30c3fd7fa53dc1067e9158e3ffc181b83a'
    audit=dict(status='pass',training_fits=len(fits),old_model_hashes_verified=len(old),prior035_preserved=True,official_cache_source_hashes_match=True,raw_cache_feature_max_error=feature_error,raw_cache_prediction_max_error=prediction_error,all_full48_records=True,record_disjoint_folds=True,no_external_or_U_training=True,new_modules_parameter_delta_squared=changes,primary=meta)
    save(WORK/'audit.json',audit);save(ROOT/'checks/structure_events_20261021.json',audit)
    # Report actual results; baseline and new structure have matched budgets.
    screen=selection['screen'];rep=selection['replicates'];means={a:float(np.mean([r['room_mean'] for r in rep if r['architecture']==a])) for a in ('baseline',v)}
    existing={(r['architecture'],r['seed']) for r in screen}
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.1%}|{r['macro_recall']:.1%}|{r['worst']:.1%}|" for r in screen+[r for r in rep if (r['architecture'],r['seed']) not in existing])
    recalls={}
    for a in ('baseline',v):
        cm=np.asarray([r['confusion_matrix'] for r in rep if r['architecture']==a]).sum(0);recalls[a]=(np.diag(cm)/cm.sum(1)).tolist()
    recalltable='\n'.join(f'|{i}人|{recalls["baseline"][i]:.1%}|{recalls[v][i]:.1%}|' for i in range(5))
    e1={a:read(WORK/'fits'/f'{a}_s42_e1/metrics.json')['validation'] for a in ('baseline',v)}
    full=read(folder/'metrics.json');maxcpu=max(c['cpu_mean_s'] for c in resource['cases']);memory=max(resource['peak_working_set_bytes'],resource['peak_commit_bytes'])
    text=f'''# 0.35反馈后的结构实验：局部事件与时延关系

上一轮外部数据ehu_radio seed42用户报告平台0.35，SHA21e7a4d…181b83a，已按确切文件登记。它没有超过两份0.37最佳。仅从他人0.52分数无法断定优势源于结构；本轮直接用同数据同训练预算检验结构。

本轮新增{len(fits)}次完整拟合，选出{v}、seed{s}，导出{resource['bytes']/1e6:.2f}MB新候选；平台未知，不替换0.37。两种子三房间均值：原版{means['baseline']:.1%}、新结构{means[v]:.1%}。这是重复使用的开发验证，不是老师平台分数，也不是新盲测。

## 为什么这样改

原0.37保留静态幅度、动态幅度、相位差、时间注意力四路，再融合分类/有序人数/存在性。前面空间CNN的下采样和末端统计会压缩局部模式，仅整段均值/标准差不必然保留扰动出现的频率。新结构是自设计适配，不复现任何论文的分数。

events替换时间路线：288个动态幅度/相位通道加3个实测时间通道，先投影128维，再并行3/9/21时点局部卷积；保留全部150时点。32种可学习软事件在五个连续片段分别提取均值、标准差、峰值、相邻变化率，并保留整段特征汇总。事件表示未接受单人、静止或速度标签，不能说每个事件对应某个人。卷积长度对应采样点数，不是精确固定秒数；时间特征显式参与，断包限制另在关系矩阵中处理。

relations新增关系路线：从原始缓存的3路真实Rx动态幅度与6路相位差，按32频组时间中心化/标准化；对每一路构造0/4/16时点延迟的32×32跨频相关矩阵，共27张图。实测时钟过滤跨长间隔配对，矩阵限幅防异常；三层小残差CNN汇总共同变化，逐通道门控制对融合潜变量的修正。相位与幅度分别保留，不假造发射/接收天线。不把相关矩阵的秩或条纹数直接当人数。与旧covariance相比，这里保留真实频率轴、直接按信号路计算且有多个延迟，不是学习后的32隐通道单一零延迟协方差。

combined使用两者。原三个空间编码器、绝对扰动强度、人数分类、有序输出、存在性约束和原损失保持。具体最终导出的是{v}，其启用模块以network.py及metadata.json为准。

### 压力失败后的单次修订

初始events的两个种子三房间均值约28.9%高于原版27.0%，但e1 seed42为0%，60个0/1窗口全部报3人，因此不能把均值提升当整体成功。该失败没有删除，原21次拟合全部保留。

随后按RESIDUAL_AMENDMENT.md新增event_residual：原时间注意力保留，额外事件编码器通过256维sigmoid门补充原时间特征，门初始-2（约11.9%），仍从头训练、同原损失和五轮预算。它增加的是事件表示而非扩大原FFN。此修订是由e1失败触发的探索，不是原先预设的独立盲验。

追加两种子各三房间及seed42 e1，优先选择e1不低于原版33.3%的方案，再比较两种子开发排名；不继续围绕e1反复搜索。实际压力门槛是否通过：{selection.get('stress_gate_passed')}。全部修订决策：`selection.json`的followup字段；若为False，本轮新候选只供结构研究，不能称整体升级。

文献依据：[CrossCount](https://arxiv.org/abs/2007.03175)关注时间遮挡模式；[DeepCount](https://arxiv.org/abs/1903.05316)结合CNN和时序，但其在线进出校正不适合本赛独立窗口；[TCN研究](https://arxiv.org/abs/1803.01271)支持局部卷积作为时序备选。已有普通LSTM/TCN/频率注意力失败，本轮没有据论文同名结构假定成功。

## 严格对照与实测

官方a1/c1/g1整房间开发，unknown不入开发，不用外部或U数据。各方案同原预处理、原class/room加权采样、5轮AdamW4e-4、weight_decay.01、batch32、原task_loss，固定末轮评估。初始化种子与采样随机源分开。seed42先四方案消融，按min(三房间均值,宏召回)、最差房间、NLL排序选新结构；再对原版与新结构seed2026复验。后续修订单列于上节。没有挑验证最高轮数，没有搜索融合权重。

|结构|种子|三房间均值|五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}

两种子合并逐类召回：

|真实人数|原版|新结构|
|---|---:|---:|
{recalltable}

e1单独压力测试（seed42，仅0/1类）：原版{e1['baseline']['accuracy']:.1%}，新结构{e1[v]['accuracy']:.1%}。它不是完整五类成绩，也不是从未分析过的房间。

最终全48记录、960窗训练池，实际抽样覆盖{full['actual_records']}记录、{full['actual_windows']}不同窗。训练数据评估{full['training_evaluation']['accuracy']:.1%}，不能解释成独立准确率。参数{meta['parameters']:,}，新模块权重有实际更新，逐例预测和混淆矩阵留D盘。

## 文件与核验

提交研究候选：`{meta['alias']}`，正式文件`{meta['path']}`，SHA256 `{meta['sha256']}`。本机真实输入、断包、全零均通过ONNX与PyTorch数值检查；CPU四线程最大用例均时{maxcpu:.3f}s，峰值约{memory/1e9:.2f}GB，ONNX18内嵌权重。只证明本机接口/资源通过，不保证老师机器耗时或平台成绩。

旧49份模型和前一版0.35文件哈希均核对未改，0.35反馈保存checks/data_revision_035_feedback.json。新候选核验见checks/structure_events_20261021.json。数据/日志/所有拟合权重留D:/huawei2026-experiments/structure_events_20261021，代码留experiments/structure_events_20261021。
'''
    (ROOT/'docs/STRUCTURE_EVENTS_20261021.md').write_text(text,encoding='utf-8');(WORK/'REPORT.md').write_text(text,encoding='utf-8')
    facts=dict(primary=meta,training_fits=len(fits),two_seed_room_means=means,recalls=recalls,prior_platform_score=.35,platform_best=.37);save(WORK/'summary.json',facts)
    registry=read(ROOT/'models/registry.json');registry['structure_events_20261021']=dict(platform_score=None,status='new_architecture_platform_unknown',primary=meta['path'],alias=meta['alias'],models=[{k:meta[k] for k in ('path','sha256','bytes','platform_score')}],report='docs/STRUCTURE_EVENTS_20261021.md',work_directory=str(WORK));save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    rows=[r for r in rows if r['path']!=meta['path']];rows.append(dict(model=f'新结构{v} seed{s}',platform_score='',path=meta['path'],sha256=meta['sha256'],source='local training only',status='new_architecture_platform_unknown',submission_id=''))
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    print('DELIVERED',json.dumps(facts,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
