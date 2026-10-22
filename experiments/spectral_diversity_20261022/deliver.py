import csv,gc,json,shutil,subprocess,sys
from pathlib import Path
import blosc2
import numpy as np
import torch
from train import ROOT,WORK,read,save,sha,load,build
class Submission(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames,intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_features(t.float(),s.float())[0]
def main():
    if (WORK/'platform_feedback_035.json').exists():
        raise RuntimeError('This historical candidate already scored 0.35; do not reset its feedback.')
    torch.set_num_threads(4);sel=read(WORK/'selection.json');item=sel['candidates'][0];v,s=item['architecture'],item['seed']
    first=read(Path('D:/huawei2026-experiments/relation_contrast_20261022/selection.json'))
    assert item['two_seed_selection']>=first['candidates'][0]['two_seed_selection'], 'Review first-stage candidate before choosing final export'
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models/best_037/model.onnx',ROOT/'models/best_038/model.onnx',ROOT/'models/best_039/model.onnx',ROOT/'models/best_039_context/model.onnx',ROOT/'models/ordinal_readout_20261021/01_context_ordinal_seed2026.onnx',ROOT/'models/spatial_research_20261021/01_pyramid_context_seed42.onnx')}
    folder=WORK/'fits'/f'{v}_s{s}_full';cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=True);model=build(v).eval();model.load_state_dict(cp['model_state']);wrapper=Submission(model).eval()
    dest=WORK/'exports'/f'{v}_s{s}';dest.mkdir(parents=True,exist_ok=True);references=[]
    for path in (ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero'):
        if path=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(path) as d:x,dt=[torch.from_numpy(d[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():z=wrapper(x,dt)
        assert z.shape==(5,) and torch.isfinite(z).all();references.append(dict(input=str(path),logits=z.tolist()))
        if len(references)==1:example=x,dt
    save(dest/'references.json',references)
    torch.onnx.export(wrapper,example,str(dest/'model.onnx'),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(dest)],check=True)
    resource=read(dest/'resource_check.json');d=load();row=d['rows'][0];a,b=int(row['start']),int(row['stop']);raw=Path('D:/huawei2026-data/labeled')
    c=blosc2.load_array(str(raw/f"CSI_{row['identifier']}.b2"));ati=blosc2.load_array(str(raw/f"ATI_{row['identifier']}.b2"));x=torch.from_numpy(c[a:b].view(np.float32).reshape(600,1,3,1024,2).copy());dt=torch.from_numpy(ati[a:b].copy());dt[-1]=0
    with torch.no_grad():
        t,sm=model.preprocess(x[None].double(),dt[None].double());fe=float((t.float()-d['t'][:1]).abs().max());pe=float((model.forward_features(t.float(),sm.float())-model.forward_features(d['t'][:1],d['s'][:1])).abs().max())
    assert fe<.001 and pe<.005
    target=ROOT/'models/spectral_diversity_20261022'/f'01_{v}_seed{s}.onnx';alias=ROOT/'submissions/spectral_diversity_20261022/model.onnx'
    for p in (target,alias):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert sha(p)==resource['sha256']
        else:shutil.copy2(dest/'model.onnx',p)
    meta=dict(architecture=v,seed=s,path=target.relative_to(ROOT).as_posix(),alias=alias.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,status='research_candidate_platform_unverified',parameters=cp['config']['parameters'],development=item,improved_development=sel['improved_development'],target_60_reached=sel['target_60_reached'],resource=resource,training=read(folder/'metrics.json'),raw_cache_feature_error=fe,raw_cache_prediction_error=pe,checkpoint=str(folder/'final.pt'))
    for p in (target.with_suffix('.json'),alias.with_suffix('.json'),WORK/'export.json'):save(p,meta)
    fits=list((WORK/'fits').glob('*/metrics.json'));assert len(fits)==15
    for p in fits:
        m=read(p);assert m['config']['epochs']==5 and m['config']['external_data'] is False
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        if m['config']['fold']=='full':assert m['actual_records']==len({r['identifier'] for r in tr})==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in va})
            assert all(r['room'] not in ('unknown',m['config']['fold']) for r in tr)
    for p,h in protected.items():assert sha(ROOT/p)==h
    save(ROOT/'checks/spectral_diversity_20261022.json',dict(status='pass',fits=15,record_disjoint=True,protected_models=protected,export=meta))
    reg=read(ROOT/'models/registry.json');assert reg['best_observed_platform_score']==.39
    reg['spectral_diversity_20261022']={k:meta[k] for k in ('path','alias','sha256','bytes','platform_score','status')};reg['recommendations']['next_submission']='spectral_diversity_20261022';reg['recommendations']['next_submission_note']='best of this relational research batch; platform unverified';save(ROOT/'models/registry.json',reg)
    path=ROOT/'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['path']!=meta['path']];rows.append(dict(model=v+' seed'+str(s),platform_score='',path=meta['path'],sha256=meta['sha256'],source='spatial architecture research; platform unverified',status=meta['status'],submission_id=''))
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    displayed=sel['controls']+sel['screen']+[r for c in sel['candidates'] for r in c['results'] if r['seed']==2026]
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|{r['worst']:.2%}|" for r in displayed)
    cm=np.asarray([r['confusion_matrix'] for r in item['results']]).sum(0);recalls=(np.diag(cm)/cm.sum(1)).tolist()
    first_fits=list(Path('D:/huawei2026-experiments/relation_contrast_20261022/fits').glob('*/metrics.json'));assert len(first_fits)==18
    for p in first_fits:
        metric=read(p)
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        if metric['config']['fold']=='full':assert metric['actual_records']==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in va})
            assert all(r['room'] not in ('unknown',metric['config']['fold']) for r in tr)
    save(ROOT/'checks/relation_first_stage_20261022.json',first)
    save(ROOT/'checks/innovation_20261022.json',dict(fits=33,record_disjoint=True,stage1=first,stage2=sel,export=meta))
    first_table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|{r['worst']:.2%}|" for r in first['screen']+[r for c in first['candidates'] for r in c['results'] if r['seed']==2026])
    report=f'''# 0.37反馈后的结构研究：谱多样性

已按SHA登记上次pyramid_context seed42平台0.37并撤回推荐。原两份0.39仍是平台最佳。本轮两阶段33次新拟合：跨通道关系18次、紧凑谱多样性15次。没有外部/U数据或旧全量权重预训练。

[协方差人数研究](https://www.mdpi.com/1424-8220/18/11/3981)使用幅度/相位协方差及扩张特征，支持研究联合变化，但其方法和实验条件不等于本题。[DeepBDC](https://arxiv.org/abs/2204.04567)研究依赖表示和少样本分类；本轮不是Brownian covariance复现。项目内新设计，未声称学术首创。

第一阶段将3路幅度/6路相位各8频段作为72节点，节点注意力读出短时与整窗关系。relation_context使用原关系，contrast_context使用短时减整窗关系并融合上下文；relation_only替换时间分支。首种子19–22%，两种子后仍未超过原对照，因此保留权重和失败记录，不优先交付。

第二阶段把协方差矩阵压缩为可解释的谱矩。幅度24节点与相位48节点分别计算原波动、3点平均、差分的整窗及5片段矩阵，以总能量、trace(C²)、trace(C³)、trace(C⁴)、参与率组成180维特征，小MLP映射到256维并门控补充原共享表示。原CNN、上下文/跨注意力、presence及分类/序数等权输出保留。只增加57216参数，不需要特征值算子。谱矩反映变化集中程度，不直接等于人数；真实房间并非简单正交变换，不能称已实现房间不变性。平滑/差分是采样尺度，不是已识别慢走/快走的人。

固定5轮AdamW4e-4、batch32、原采样/loss，三房间整房间留出，两个种子复验，e1压力测试；每阶段只全量训练最佳一份。min(房间均值,宏召回)排序，不挑epoch。所有统计与变换仅在单个输入内计算，不使用验证集总体统计。

|结构|种子|三房间均值|五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}
{first_table}

最终候选{v} seed{s}，两种子三房间均值{item['two_seed_room_mean']:.2%}；超过两份原对照开发指标：{sel['improved_development']}；达到60%：{sel['target_60_reached']}。合并0—4人数召回{recalls}；e1(seed42)：{item['e1']}。开发集反复使用，不能证明统计显著或平台提高。

有标签数据每布局一房间，因此本地留房间也留布局。全量实际覆盖48记录、{meta['training']['actual_windows']}/960不同窗口；训练数据重评不是独立测试。平台最佳仍0.39，新候选未测。

谱统计诊断见checks/spectral_feature_diagnosis_20261022.json：a1幅度参与率按0—4人均值约0.356/0.275/0.211/0.167/0.090，c1的1—4人约0.248/0.202/0.224/0.204。既不是人数增加时必然上升，也不是跨房间统一单调，因此不能直接数特征值或强制单调。每组只有少量独立记录，窗口均值不是独立实验。这个观察用于解释限制，不用于改变预设选择标准。

正式文件{meta['path']}，提交副本{meta['alias']}；{meta['bytes']}字节，SHA256 {meta['sha256']}。ONNX接口/嵌入权重/三种输入数值检查通过；本机4线程最慢用例平均{max(r['cpu_mean_s'] for r in resource['cases']):.4f}s、峰值提交内存{resource['peak_commit_bytes']/1e9:.3f}GB；不保证平台机器速度。原始输入与缓存预测差{pe:.6g}。

全部权重、预测、清单位于D:/huawei2026-experiments/relation_contrast_20261022及spectral_diversity_20261022。原0.37/0.38/两份0.39/序数0.38/金字塔0.37文件受哈希核验保护。
'''
    (ROOT/'docs/SPECTRAL_DIVERSITY_20261022.md').write_text(report,encoding='utf-8');(WORK/'REPORT.md').write_text(report,encoding='utf-8')
    source_files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src').glob('*.py'))+[ROOT/'experiments/global_local_20261021/network.py',ROOT/'experiments/structure_events_20261021/train.py']
    save(WORK/'source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in source_files})
    phase1=Path('D:/huawei2026-experiments/relation_contrast_20261022')
    (phase1/'REPORT.md').write_text('本阶段未导出ONNX，不优先提交。最终联合报告见../spectral_diversity_20261022/REPORT.md。\n\n'+report,encoding='utf-8')
    for p in (ROOT/'experiments/relation_contrast_20261022').iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,phase1/'source_snapshot'/p.name)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    print('DELIVERED',json.dumps({k:meta[k] for k in ('architecture','seed','path','bytes','sha256','improved_development','target_60_reached')}),flush=True)
if __name__=='__main__':main()
