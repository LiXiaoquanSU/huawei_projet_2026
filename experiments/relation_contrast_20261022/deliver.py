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
    torch.set_num_threads(4);sel=read(WORK/'selection.json');item=sel['candidates'][0];v,s=item['architecture'],item['seed']
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
    target=ROOT/'models/relation_contrast_20261022'/f'01_{v}_seed{s}.onnx';alias=ROOT/'submissions/relation_contrast_20261022/model.onnx'
    for p in (target,alias):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert sha(p)==resource['sha256']
        else:shutil.copy2(dest/'model.onnx',p)
    meta=dict(architecture=v,seed=s,path=target.relative_to(ROOT).as_posix(),alias=alias.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,status='research_candidate_platform_unverified',parameters=cp['config']['parameters'],development=item,improved_development=sel['improved_development'],target_60_reached=sel['target_60_reached'],resource=resource,training=read(folder/'metrics.json'),raw_cache_feature_error=fe,raw_cache_prediction_error=pe,checkpoint=str(folder/'final.pt'))
    for p in (target.with_suffix('.json'),alias.with_suffix('.json'),WORK/'export.json'):save(p,meta)
    fits=list((WORK/'fits').glob('*/metrics.json'));assert len(fits)==18
    for p in fits:
        m=read(p);assert m['config']['epochs']==5 and m['config']['external_data'] is False
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        if m['config']['fold']=='full':assert m['actual_records']==len({r['identifier'] for r in tr})==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in va})
            assert all(r['room'] not in ('unknown',m['config']['fold']) for r in tr)
    for p,h in protected.items():assert sha(ROOT/p)==h
    save(ROOT/'checks/relation_contrast_20261022.json',dict(status='pass',fits=18,record_disjoint=True,protected_models=protected,export=meta))
    reg=read(ROOT/'models/registry.json');assert reg['best_observed_platform_score']==.39
    reg['relation_contrast_20261022']={k:meta[k] for k in ('path','alias','sha256','bytes','platform_score','status')};reg['recommendations']['next_submission']='relation_contrast_20261022';reg['recommendations']['next_submission_note']='best of this relational research batch; platform unverified';save(ROOT/'models/registry.json',reg)
    path=ROOT/'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['path']!=meta['path']];rows.append(dict(model=v+' seed'+str(s),platform_score='',path=meta['path'],sha256=meta['sha256'],source='spatial architecture research; platform unverified',status=meta['status'],submission_id=''))
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    displayed=sel['controls']+sel['screen']+[r for c in sel['candidates'] for r in c['results'] if r['seed']==2026]
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|{r['worst']:.2%}|" for r in displayed)
    cm=np.asarray([r['confusion_matrix'] for r in item['results']]).sum(0);recalls=(np.diag(cm)/cm.sum(1)).tolist()
    report=f'''# 跨通道关系对比结构实验

上轮多层频率特征模型pyramid_context seed42平台0.37，已按SHA登记并撤回推荐。原两份0.39仍是平台最佳。本轮完成18次新拟合，无外部/U/旧权重预训练。

结构与论文边界见experiments/relation_contrast_20261022/PLAN.md。借鉴[DeepBDC](https://arxiv.org/abs/2204.04567)对依赖关系表示的关注，但采用普通相关系数，不是论文复现；[WiCount-DASL](https://research.polyu.edu.hk/en/publications/wicount-dasl-domain-adversarial-semi-supervised-learning-for-wi-f/)仅核对摘要，不宣称复现其域对齐。本设计是项目内结构创新，不声称学术首创。

将3路动态幅度、6路相位各分8个频带，构成72节点。计算整窗和五个片段的跨节点相关关系，保留能量与均值；通过节点注意力、片段注意力汇总。relation_context使用局部相关矩阵，contrast_context使用局部减整窗相关，两者都保留整窗关系并与原上下文分支门控融合。relation_only用差异关系替换原时间分支。保留原静态/动态/相位CNN，不把平稳分量称为空房背景，不把节点称为人。原始与差异关系线性可互推，检验的是归纳偏置而非新增信息。

固定5轮、AdamW4e-4、batch32、原抽样/loss。seed42三折筛选9次，前二seed2026三折6次、e1两次、全量一次。按min(三房间均值,宏召回)选择，固定末轮，不挑epoch。原对照复用并复算context2026a1。

|结构|种子|三房间均值|五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}

选择{v} seed{s}，两种子三房间均值{item['two_seed_room_mean']:.2%}；预设开发指标超过原两种0.39结构：{sel['improved_development']}；60%达到：{sel['target_60_reached']}。合并0—4人数召回{recalls}。e1(seed42)：{item['e1']}。开发集反复使用，两种子不足以证明显著性，平台未知。

每种已知布局只有一个房间，因此留房间也留布局；不能把此分数当平台成绩。全量训练实际覆盖48记录、{meta['training']['actual_windows']}/960不同窗口；全量training_evaluation不是独立测试。

正式文件{meta['path']}；提交副本{meta['alias']}。{meta['bytes']}字节，SHA256 {meta['sha256']}。ONNX嵌入权重、接口、三个输入数值及CPU资源检查通过，本机4线程最慢用例平均{max(r['cpu_mean_s'] for r in resource['cases']):.4f}s，峰值提交内存{resource['peak_commit_bytes']/1e9:.3f}GB；不保证平台速度。原始输入/缓存预测误差{pe:.6g}。

原模型哈希保持，实验文件在D:/huawei2026-experiments/relation_contrast_20261022。新候选平台未测，原最佳仍0.39。
'''
    (ROOT/'docs/RELATION_CONTRAST_20261022.md').write_text(report,encoding='utf-8');(WORK/'REPORT.md').write_text(report,encoding='utf-8')
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    print('DELIVERED',json.dumps({k:meta[k] for k in ('architecture','seed','path','bytes','sha256','improved_development','target_60_reached')}),flush=True)
if __name__=='__main__':main()
