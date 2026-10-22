import gc,json,shutil,subprocess,sys
from pathlib import Path
import blosc2
import numpy as np
import torch
from common import ROOT,WORK,RAW,FeatureStore,build,read,save,sha

class Submission(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames,intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_features(t.float(),s.float())[0]

def export(item,number,store):
    name,recipe,seed=item['architecture'],item['recipe'],item['seed'];folder=WORK/'fits'/f'{name}_{recipe}_s{seed}_full';cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    model=build(name).eval();model.load_state_dict(cp['model_state']);wrapper=Submission(model).eval();dest=WORK/'exports'/f'{name}_{recipe}_s{seed}';dest.mkdir(parents=True,exist_ok=True);refs=[]
    for path in (ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero'):
        if path=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(path) as d:x,dt=[torch.from_numpy(d[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():p=wrapper(x,dt)
        assert p.shape==(5,) and torch.isfinite(p).all();refs.append(dict(input=str(path),logits=p.tolist()))
        if len(refs)==1:example=x,dt
    save(dest/'references.json',refs)
    torch.onnx.export(wrapper,example,str(dest/'model.onnx'),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(dest)],check=True)
    resource=read(dest/'resource_check.json')
    # Verify a newly-created offset window, not merely one of the old anchors.
    i=next(i for i,r in enumerate(store.rows) if not r['baseline']);row=store.rows[i];a,b=row['start'],row['stop']
    c=blosc2.load_array(str(RAW/f"CSI_{row['identifier']}.b2"));ati=blosc2.load_array(str(RAW/f"ATI_{row['identifier']}.b2"));x=torch.from_numpy(c[a:b].view(np.float32).reshape(600,1,3,1024,2).copy());dt=torch.from_numpy(ati[a:b].copy());dt[-1]=0;cached_t,cached_s=store.get(i)
    with torch.no_grad():
        t,s=model.preprocess(x[None].double(),dt[None].double());fe=float((t.float()[0]-cached_t).abs().max());pe=float((model.forward_features(t.float(),s.float()).softmax(-1)-model.forward_features(cached_t[None],cached_s[None]).softmax(-1)).abs().max())
    assert fe<.001 and pe<.005
    target=ROOT/'models/sliding_windows_20261022'/f'{number:02d}_{name}_{recipe}_seed{seed}.onnx';alias=ROOT/'submissions/sliding_windows_20261022'/f'{number:02d}_{name}.onnx'
    for p in (target,alias):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert sha(p)==resource['sha256']
        else:shutil.copy2(dest/'model.onnx',p)
    meta=dict(architecture=name,recipe=recipe,seed=seed,path=target.relative_to(ROOT).as_posix(),alias=alias.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=resource['bytes'],parameters=cp['config']['parameters'],platform_score=None,status='overlapping_window_research_candidate_platform_unverified',development=item,training=read(folder/'metrics.json'),sampling_audit=read(folder/'sampling_audit.json'),resource=resource,new_offset_cache_feature_error=fe,new_offset_probability_error=pe,checkpoint=str(folder/'final.pt'))
    for p in (target.with_suffix('.json'),alias.with_suffix('.json'),dest/'export.json'):save(p,meta)
    del model,wrapper,cp,c,ati;gc.collect();print('EXPORTED',name,meta['bytes'],meta['sha256'],flush=True);return meta

def main():
    torch.set_num_threads(4);store=FeatureStore();sel=read(WORK/'selection.json');assert read(WORK/'cache/audit.json')['status']=='pass'
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models/best_037/model.onnx',ROOT/'models/best_038/model.onnx',ROOT/'models/best_039/model.onnx',ROOT/'models/best_039_context/model.onnx',ROOT/'models/fresh_backbones_20261022/01_mixer_seed42.onnx')}
    fits=list((WORK/'fits').glob('*/metrics.json'));assert len(fits)==33
    for p in fits:
        m=read(p);cfg=m['config'];assert cfg['initialization']=='fresh' and cfg['external_data'] is False
        assert cfg['optimizer_steps_per_epoch']==(cfg['samples_per_epoch']+31)//32
        import csv
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        if cfg['fold']=='full':assert m['actual_records']==len({r['identifier'] for r in tr})==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            with (p.parent/'dense_validation_manifest.csv').open(encoding='utf-8-sig') as f:dense=list(csv.DictReader(f))
            assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in dense})
            assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in va})
            assert all(r['room'] not in ('unknown',cfg['fold']) for r in tr)
            assert all(r['baseline']=='True' for r in va)
    metas=[export(item,k,store) for k,item in enumerate(sel['candidates'],1)]
    top=metas[0];alias=ROOT/'submissions/sliding_windows_20261022/model.onnx';shutil.copy2(ROOT/top['path'],alias);save(alias.with_suffix('.json'),top)
    for p,h in protected.items():assert sha(ROOT/p)==h
    save(WORK/'exports.json',metas);save(ROOT/'checks/sliding_training_20261022.json',dict(status='pass',fits=33,record_disjoint=True,baseline_validation_unchanged=True,update_count_matched=True,protected_models=protected,exports=metas))
    registry=read(ROOT/'models/registry.json');assert registry['best_observed_platform_score']==.39
    registry['sliding_windows_20261022']=dict(platform_score=None,status='three_existing_architectures_retrained_with_overlapping_windows',candidates=[{k:m[k] for k in ('path','alias','sha256','bytes','platform_score','architecture','recipe')} for m in metas])
    registry['recommendations']['next_submission']='sliding_windows_20261022';registry['recommendations']['next_submission_note']='ranked by unchanged primary validation after overlapping-window data expansion; platform unverified';save(ROOT/'models/registry.json',registry)
    scorefile=ROOT/'models/platform_scores.csv'
    with scorefile.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    paths={m['path'] for m in metas};rows=[r for r in rows if r['path'] not in paths]
    for m in metas:rows.append(dict(model=f"{m['architecture']} {m['recipe']} seed{m['seed']}",platform_score='',path=m['path'],sha256=m['sha256'],source='overlapping-window retraining; platform unverified',status=m['status'],submission_id=''))
    with scorefile.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    table='\n'.join(f"|{m['architecture']}|{m['recipe']}|{m['development']['baseline_room_mean']:.2%}|{m['development']['two_seed_room_mean']:.2%}|{np.mean([r['dense_record_room_mean'] for r in m['development']['results']]):.2%}|{m['bytes']/1e6:.2f}|" for m in metas)
    screen='\n'.join(f"|{r['architecture']}|{r['recipe']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|" for r in sel['screen'])
    counts=store.index['counts'];report=f'''# 重叠窗口数据扩展与三模型重训

用户要求暂停结构探索，转向老师给的连续CSI数据。本轮不改三份网络结构：cross_events/context_events平台0.39，MLP Mixer平台0.33。时频卷积结构研究结果保留，不继续该路线。

48条原始记录各12000帧，每窗仍600帧，旧不重叠窗口{counts['baseline']}；按ATI累计真实时间每1秒及0.5秒起窗，并保留旧锚点后分别{counts['step_1s']}、{counts['step_0p5s']}。独立记录仍48，没有新增房间或人数场景。前者约为原窗数{counts['step_1s']/960:.2f}倍，后者{counts['step_0p5s']/960:.2f}倍；不是独立数据量增长倍数。未增加外部或无标签U数据。

每条记录、每个房间作为整体分配，unknown不进开发；训练与验证源ID无交叉，因此不可能有同一帧跨训练/验证的重叠。人数标签来自原CSV，每窗不跨记录、最后ATI置0。旧960锚点特征逐位复用，旧三份seed42a1权重预测复算一致。新增窗从原始CSI高精度预处理，数值差异经锚点/预测核对；缓存审计见checks/sliding_integrity_20261022.json。

主要测试保持原不重叠窗口，不用密集窗口更换旧成绩分母。另扩展独立留出记录的0.5s密集测试，报告按记录平均精度与相邻一致率（一直判错也可能一致）。不能将重叠窗口当独立实验或扩大置信度。每布局仍只有一个房间，留房间同时留布局。

固定更新预算：每epoch抽样数仍为原训练折非重叠窗数；原类别/房间权重并对源记录等权，避免较长记录因新窗更多而放大。cross/context仍5轮原复合loss，Mixer仍12轮平滑CE，AdamW4e-4、batch32、wd.01不变，均从头初始化。新窗可选池更大，实际遍历数另列；并未声称所有新窗均被抽到。

3模型×2步长×3房间seed42=18次，每模型最佳步长seed2026三折9次、e1三次、全48记录三次，共33次。固定末轮，按min(主测试三房间均值,宏召回)选步长。最终种子预先固定匹配已测源文件cross42/context2026/mixer42，不按本地高分挑种子。

|模型|选中数据版本|原两种子主测试|重叠训练后主测试|密集测试按记录均值|大小MB|
|---|---|---:|---:|---:|---:|
{table}

首种子筛选明细：

|模型|数据版本|种子|主测试均值|宏召回|
|---|---|---:|---:|---:|
{screen}

三模型各人数召回、e1、实际新窗覆盖及资源检查都在各模型同名JSON。开发反复使用，两种子不足以证明统计显著或平台改善。全量training_evaluation是已参与训练记录的旧锚点评估，不是独立测试。

本轮排序第一为{top['path']}，统一副本submissions/sliding_windows_20261022/model.onnx。共三份候选，均嵌入权重、输入600帧、输出5类、三个输入数值/CPU资源检查通过；平台分数未知，不继承0.39/0.33。平台最佳仍两份0.39，旧文件哈希保护。权重、缓存、源记录清单、逐例预测和审计在D:/huawei2026-experiments/sliding_windows_20261022。
'''
    (ROOT/'docs/SLIDING_WINDOWS_20261022.md').write_text(report,encoding='utf-8');(WORK/'REPORT.md').write_text(report,encoding='utf-8')
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    save(WORK/'source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src').glob('*.py'))})
    print('DELIVERED',[(m['path'],m['bytes']) for m in metas],flush=True)
if __name__=='__main__':main()
