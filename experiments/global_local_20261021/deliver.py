"""Export two ranked research models; protect the platform-confirmed .38."""
import csv,gc,json,shutil,subprocess,sys
from pathlib import Path
import blosc2
import numpy as np
import torch
from train import ROOT,WORK,OLD,read,save,sha,seed,load,build

class Submission(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames,intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_features(t.float(),s.float())[0]

def export(item,number,d):
    v,s=item['architecture'],item['seed'];folder=WORK/'fits'/f'{v}_s{s}_full';cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    model=build(v).eval();model.load_state_dict(cp['model_state']);wrapper=Submission(model).eval();dest=WORK/'exports'/f'{v}_s{s}';dest.mkdir(parents=True,exist_ok=True);references=[]
    for path in [ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero']:
        if path=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(path) as data:x,dt=[torch.from_numpy(data[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():z=wrapper(x,dt)
        assert z.shape==(5,) and torch.isfinite(z).all();references.append(dict(input=str(path),logits=z.tolist()))
        if len(references)==1:example=x,dt
    save(dest/'references.json',references)
    torch.onnx.export(wrapper,example,str(dest/'model.onnx'),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(dest)],check=True)
    resource=read(dest/'resource_check.json');target=ROOT/'models/global_local_20261021'/f'{number:02d}_{v}_seed{s}.onnx';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==resource['sha256']
    else:shutil.copy2(dest/'model.onnx',target)
    # Check training-cache path and raw-input path agree for this new network.
    row=d['rows'][0];a,b=int(row['start']),int(row['stop']);raw=Path('D:/huawei2026-data/labeled')
    c=blosc2.load_array(str(raw/f"CSI_{row['identifier']}.b2"));ati=blosc2.load_array(str(raw/f"ATI_{row['identifier']}.b2"))
    x=torch.from_numpy(c[a:b].view(np.float32).reshape(600,1,3,1024,2).copy());dt=torch.from_numpy(ati[a:b].copy());dt[-1]=0
    with torch.no_grad():
        t,sm=model.preprocess(x[None].double(),dt[None].double());t,sm=t.float(),sm.float();fe=float((t-d['t'][:1]).abs().max());pe=float((model.forward_features(t,sm)-model.forward_features(d['t'][:1],d['s'][:1])).abs().max())
    assert fe<.001 and pe<.005,(fe,pe)
    seed(s);initial=build(v).state_dict();delta=sum(float((value-initial[k]).square().sum()) for k,value in cp['model_state'].items() if k.startswith('temporal.'));assert delta>0
    meta=dict(architecture=v,seed=s,path=target.relative_to(ROOT).as_posix(),bytes=resource['bytes'],sha256=resource['sha256'],parameters=cp['config']['parameters'],platform_score=None,status='research_candidate_platform_unknown',config=cp['config'],development=item,resource=resource,checkpoint=str(folder/'final.pt'),raw_cache_feature_max_error=fe,raw_cache_prediction_max_error=pe,temporal_parameter_delta_squared=delta,external_data=False,unlabeled_data=False,initialization='random, no full-data checkpoint or external pretraining')
    save(target.with_suffix('.json'),meta);print('EXPORTED',number,v,meta['bytes'],meta['sha256'],flush=True)
    del model,wrapper,cp,initial,c,ati;gc.collect();return meta

def main():
    if (WORK/'platform_feedback_039.json').exists():raise RuntimeError('Cross-events is now platform0.39; use a new experiment directory rather than overwriting feedback.')
    torch.set_num_threads(4);selection=read(WORK/'selection.json');d=load();registry=read(ROOT/'models/registry.json');assert registry['best_observed_platform_score']==.38
    protected={name:sha(ROOT/name) for name in ['models/best_038/model.onnx','models/structure_events_20261021/01_events_seed42.onnx','submissions/structure_events_20261021/model.onnx','models/best_037/model.onnx','models/random_restart_20261020/02_original_seed42.onnx']}
    assert protected['models/best_038/model.onnx']=='6d3799c7179b01f0cb9c36194aa8f60d6709b4851db82290bdb60db1143a06e3'
    results=[export(item,n,d) for n,item in enumerate(selection['candidates'],1)]
    alias=ROOT/'submissions/global_local_20261021';alias.mkdir(parents=True,exist_ok=True)
    for n,m in enumerate(results):
        target=alias/('model.onnx' if n==0 else 'alternative_model.onnx')
        if target.exists():assert sha(target)==m['sha256']
        else:shutil.copy2(ROOT/m['path'],target)
        save(target.with_suffix('.json'),m)
    save(alias/'metadata.json',results[0]);save(WORK/'export_registry.json',results);save(WORK/'primary_candidate.json',results[0])
    fits=[]
    for p in sorted((WORK/'fits').glob('*/metrics.json')):
        m=read(p)
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        ids={r['identifier'] for r in tr};config=m['config'];assert config['external_data'] is False and config['epochs']==5
        if config['fold']=='full':assert len(ids)==m['actual_records']==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            assert ids.isdisjoint({r['identifier'] for r in va}) and all(r['room'] not in ('unknown',config['fold']) for r in tr)
        fits.append(m)
    assert len(fits)==19,len(fits)
    for name,expected in protected.items():assert sha(ROOT/name)==expected
    for name,expected in read(ROOT/'checks/restarts_cleanup_20261020.json')['model_hashes'].items():assert sha(ROOT/name)==expected
    audit=dict(status='pass',complete_new_fits=len(fits),old49_model_hashes_preserved=True,protected_platform_models=protected,training_record_disjoint=True,full48_records_covered=True,no_external_U_or_pretraining=True,control_identity=read(WORK/'reused_controls.json'),exports=results,invalid_axis_run_excluded=True)
    save(WORK/'audit.json',audit);save(ROOT/'checks/global_local_20261021.json',audit)
    controls=read(WORK/'reused_controls.json')['results'];screen=selection['screen'];rep=[r for item in selection['candidates'] for r in item['results']]
    displayed=controls+screen+[r for r in rep if r['seed']==2026]
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.1%}|{r['macro_recall']:.1%}|{r['worst']:.1%}|{r['rooms']['a1']:.1%}|{r['rooms']['c1']:.1%}|{r['rooms']['g1']:.1%}|" for r in displayed)
    recalls={};means={}
    for v in ['baseline','events']+[m['architecture'] for m in results]:
        rr=[r for r in controls+rep if r['architecture']==v];cm=np.asarray([r['confusion_matrix'] for r in rr]).sum(0);recalls[v]=(np.diag(cm)/cm.sum(1)).tolist();means[v]=float(np.mean([r['room_mean'] for r in rr]))
    headers=list(recalls);recalltable='|真实人数|'+'|'.join(headers)+'|\n|---|'+'---:|'*len(headers)+'\n'+'\n'.join('|'+str(c)+'人|'+'|'.join(f'{recalls[v][c]:.1%}' for v in headers)+'|' for c in range(5))
    e1={v:read(OLD/'fits'/f'{v}_s42_e1/metrics.json')['validation'] for v in ('baseline','events')};e1.update({m['architecture']:m['development']['e1'] for m in results})
    e1table='\n'.join(f"|{v}|{m['accuracy']:.1%}|{sum(sum(row[2:]) for row in m['confusion_matrix'][:2])}/60|" for v,m in e1.items())
    modeltable='\n'.join(f"|{n}|{m['architecture']} seed{m['seed']}|{m['bytes']/1e6:.2f}MB|{max(c['cpu_mean_s'] for c in m['resource']['cases']):.3f}s|" for n,m in enumerate(results,1))
    invalid=list((WORK/'_invalid_segment_axis_v0').glob('fits/*/metrics.json'))
    complement=read(WORK/'complementarity.json')['totals']
    text=f'''# 0.37与0.38之后：全局与局部事件结构实验

平台已确认最高仍0.38，文件SHA6d3799c7…143a06e3。此次新候选平台未测，不按本地结果改平台排名。新增19次完整拟合（9次三结构筛选、6次第二种子、2次e1、2次全量）。baseline/events历史两种子与e1复用原结果，不算新训练；seed42a1预测重算与原混淆矩阵一致，原代码/检查点身份存reused_controls.json。

## 实验假设与结构

0.37具有75个时间token的整段注意力，0.38具有多尺度局部事件统计；上轮事件向量门控加到全局向量没有稳定收益。这轮只修改第四条时间路线，保留三个空间CNN及融合/有序人数/绝对强度/原loss。

配对误差分析：两个种子对同一820个物理留出窗口形成1640次对应评估，只有原版正确{complement['only_baseline_correct']}次、只有事件版正确{complement['only_events_correct']}次，两者都错{complement['both_wrong']}次。可见存在互补，也有大量共同失败；同一窗口在不同种子下重复评估不算新独立样本。诊断中的真值择优oracle不能部署、不是平台预估，未搜索融合权重。

context_events把0.38的五段事件统计变成5个128维token，加入实测时钟编码，两个注意力块学习前后事件关系，再与整段局部卷积汇总投影256维。共享片段编码可减少固定时间位置的参数依赖；这是设计假设，不保证环境不变。

cross_events保留0.37时间注意力的75个256维token，以五个局部事件token为query、全局token为key/value，8头交叉注意力交换信息，再做片段上下文注意力和256维汇总。不同于上轮两个最终256维向量相加；也不是把0.37与0.38两份已评测ONNX直接融合。整体从头训练。

band_events把32频组划成4个连续频带，每带通过共享3/9/21时点局部卷积和软事件，构成4带×5段的20个128维token，使用实际时间与频带坐标学习关联。没有插值新频点，也不把事件当逐人检测。

依据：[CrossViT](https://arxiv.org/abs/2103.14899)的跨尺度信息交换、[InceptionTime](https://arxiv.org/abs/1909.04939)的多尺度卷积；只是建模动机，本轮是CSI自设计适配，不引用论文图像/时序成绩来预测赛题效果。

## 数据、训练与选择

只有比赛有标签数据。整房间开发排除对应验证房间与unknown；最终全48记录的960窗口池。无外部、无U、无预训练、无全量最佳权重的验证初始化。五轮AdamW4e-4、weight decay.01、batch32、梯度裁剪3，原loss/原class-room采样；相同训练循环源码经哈希记录。固定末轮，不按验证峰值选轮。

三结构seed42按min(三房间均值,五类宏召回)、最差房间、NLL选两项；两种子复验后按平均同指标排序，选对应开发较强种子做各自全量训练。e1继续报告而不是硬门槛：它只有0/1且已反复分析，历史events e1为0但平台最高0.38，不能用单一失败绝对淘汰新模型。本地开发已经多轮使用，不是盲测，两个种子不能证明统计显著提升。

|结构|种子|三房间均值|宏召回|最差房间|a1|c1|g1|
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

两种子三房间均值：{', '.join(v+' '+format(a,'.1%') for v,a in means.items())}。

## 逐类与压力结果

{recalltable}

这些召回按两个种子合并混淆矩阵计算，4人真实样本非空。变化可能伴随类别间取舍，不能只看平均分。

|seed42 e1方案|0/1准确率|错误报2—4人的窗口|
|---|---:|---:|
{e1table}

e1与完整五类测试不同，不把其准确率直接填为平台预估。实际混淆矩阵、逐例预测、每折记录清单、历史/参数都在D盘fits目录。

## 提交研究候选与检查

|本地顺序|结构|大小|本机最慢用例均时|
|---|---|---:|---:|
{modeltable}

首选研究副本：submissions/global_local_20261021/model.onnx；备选副本：同目录alternative_model.onnx。正式带结构名的文件在models/global_local_20261021/，身份以JSON与SHA为准。两者均平台未知；0.38最佳副本不替换。

各候选全量训练实际覆盖全部48记录，训练池960窗与有放回采样实际覆盖的不同窗口分开记在metrics.json。训练数据准确率不是独立泛化成绩。原始输入到特征缓存预测一致性、固定float32接口、真实/断包/零输入、ONNX18嵌入权重、<100MB/<.7s/<6GB本机检查均通过；不保证老师机器耗时或分数。新时间模块参数有实际更新。

启动复查发现片段统计拼接轴错误后，停止指定训练并归档所有错误输出到_invalid_segment_axis_v0；其中{len(invalid)}次有末轮指标，不参与选择，也不算正式19次。修正后用各段恒定且互异的输入检查片段身份，三个模块形状/有限值/梯度都通过，然后全部正式重跑。错误与修复不隐藏，也不据错误分数挑方案。

旧49份ONNX及0.37/0.38正式/副本哈希全部核对未变。轻量审计checks/global_local_20261021.json；原始日志、权重、源码快照在D:/huawei2026-experiments/global_local_20261021；新代码experiments/global_local_20261021。平台反馈后按确切文件登记，不把本地成绩冒充新高分。
'''
    (ROOT/'docs/GLOBAL_LOCAL_20261021.md').write_text(text,encoding='utf-8');(WORK/'REPORT.md').write_text(text,encoding='utf-8');save(WORK/'summary.json',dict(new_fits=19,platform_best=.38,means=means,recalls=recalls,exports=results))
    registry['global_local_20261021']=dict(platform_score=None,status='research_candidates_platform_unknown',primary=results[0]['path'],alias='submissions/global_local_20261021/model.onnx',models=[{k:m[k] for k in ('path','sha256','bytes','platform_score')} for m in results],report='docs/GLOBAL_LOCAL_20261021.md',work_directory=str(WORK));assert registry['best_observed_platform_score']==.38;save(ROOT/'models/registry.json',registry)
    path=ROOT/'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if not r['path'].startswith('models/global_local_20261021/')]
    rows.extend(dict(model=m['architecture']+' seed'+str(m['seed']),platform_score='',path=m['path'],sha256=m['sha256'],source='local structure research only',status='research_candidate_platform_unknown',submission_id='') for m in results)
    with path.open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    snap=WORK/'source_snapshot'
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    shutil.copy2(ROOT/'experiments/structure_events_20261021/train.py',snap/'reference_train.py');shutil.copy2(ROOT/'experiments/structure_events_20261021/network.py',snap/'reference_event_network.py')
    core=snap/'core';core.mkdir(exist_ok=True)
    for p in (ROOT/'src').glob('*.py'):shutil.copy2(p,core/p.name)
    print('DELIVERED_TWO_CANDIDATES',json.dumps(dict(means=means,paths=[m['path'] for m in results],sizes=[m['bytes'] for m in results],new_fits=19,platform_best=.38),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
