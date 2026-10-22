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
    if (WORK/'platform_feedback_033.json').exists():
        raise RuntimeError('Mixer already evaluated at 0.33; do not reset historical feedback.')
    torch.set_num_threads(4);sel=read(WORK/'selection.json');item=sel['candidates'][0];v,s=item['architecture'],item['seed']
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models/best_037/model.onnx',ROOT/'models/best_038/model.onnx',ROOT/'models/best_039/model.onnx',ROOT/'models/best_039_context/model.onnx',ROOT/'models/ordinal_readout_20261021/01_context_ordinal_seed2026.onnx',ROOT/'models/spatial_research_20261021/01_pyramid_context_seed42.onnx',ROOT/'models/spectral_diversity_20261022/01_diversity_context_seed42.onnx')}
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
    target=ROOT/'models/fresh_backbones_20261022'/f'01_{v}_seed{s}.onnx';alias=ROOT/'submissions/fresh_backbones_20261022/model.onnx'
    for p in (target,alias):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert sha(p)==resource['sha256']
        else:shutil.copy2(dest/'model.onnx',p)
    meta=dict(architecture=v,seed=s,path=target.relative_to(ROOT).as_posix(),alias=alias.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,status='research_candidate_platform_unverified',parameters=cp['config']['parameters'],development=item,improved_development=sel['improved_development'],target_60_reached=sel['target_60_reached'],resource=resource,training=read(folder/'metrics.json'),raw_cache_feature_error=fe,raw_cache_prediction_error=pe,checkpoint=str(folder/'final.pt'))
    history_root=Path('D:/huawei2026-experiments/global_local_20261021/fits')
    historical={name:float(np.mean([read(history_root/f'{name}_s{seed}_{fold}/metrics.json')['validation']['accuracy'] for seed in (42,2026) for fold in ('a1','c1','g1')])) for name in ('context_events','cross_events')}
    meta['historical_reference_means']=historical;meta['exceeds_historical_room_mean']=item['two_seed_room_mean']>max(historical.values())
    for p in (target.with_suffix('.json'),alias.with_suffix('.json'),WORK/'export.json'):save(p,meta)
    fits=list((WORK/'fits').glob('*/metrics.json'));assert len(fits)==24
    for p in fits:
        m=read(p);assert m['config']['epochs']==12 and m['config']['external_data'] is False
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        if m['config']['fold']=='full':assert m['actual_records']==len({r['identifier'] for r in tr})==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in va})
            assert all(r['room'] not in ('unknown',m['config']['fold']) for r in tr)
    for p,h in protected.items():assert sha(ROOT/p)==h
    save(ROOT/'checks/fresh_backbones_20261022.json',dict(status='pass',fits=24,record_disjoint=True,protected_models=protected,export=meta))
    reg=read(ROOT/'models/registry.json');assert reg['best_observed_platform_score']==.39
    reg['fresh_backbones_20261022']={k:meta[k] for k in ('path','alias','sha256','bytes','platform_score','status')};reg['recommendations']['next_submission']='fresh_backbones_20261022';reg['recommendations']['next_submission_note']='best of independent backbone research batch; platform unverified';save(ROOT/'models/registry.json',reg)
    path=ROOT/'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['path']!=meta['path']];rows.append(dict(model=v+' seed'+str(s),platform_score='',path=meta['path'],sha256=meta['sha256'],source='independent backbone research; platform unverified',status=meta['status'],submission_id=''))
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    displayed=sel['controls']+sel['screen']+[r for c in sel['candidates'] for r in c['results'] if r['seed']==2026]
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|{r['worst']:.2%}|" for r in displayed)
    cm=np.asarray([r['confusion_matrix'] for r in item['results']]).sum(0);recalls=(np.diag(cm)/cm.sum(1)).tolist()
    random_root=Path('D:/huawei2026-experiments/random_dictionary_20261022');random_result=read(random_root/'selection.json')
    assert item['two_seed_selection']>random_result['candidates'][0]['two_seed_selection']
    random_fits=list((random_root/'fits').glob('*/metrics.json'));assert len(random_fits)==8
    for p in random_fits:
        m=read(p);assert m['config']['ridge_lambda']==10
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
        assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in va})
        assert all(r['room'] not in ('unknown',m['config']['fold']) for r in tr)
    save(ROOT/'checks/independent_research_20261022.json',dict(fits=32,record_disjoint=True,neural=sel,random_dictionary=random_result,export=meta))
    report=f'''# 从头设计的独立骨干实验

上一谱多样性模型已按SHA登记平台0.35。原两份0.39仍是平台最佳。金字塔0.37、序数读出0.38、谱多样性0.35没有支持不断给原模型加分支；这次改为独立骨干。

三个新模型都不继承原StructuredModel骨干，不使用原空间CNN/专家混合/presence/序数头/signal辅助头，不加载旧权重。共享的只有确定性CSI预处理与数据划分。幅度128频点额外均值降到32，与相位32频点对齐形成12×32×150输入，因此也是新的输入组织/单头训练方案，不能把所有变化归因于单个结构。

band_patch：8个频带共享3层96维Transformer，对每带10个15点patch编码后集合池化，弱化绝对频段身份。mixer：30个5点patch，4层时间/特征交替MLP，没有卷积/注意力骨干。bigru：两层双向128维GRU，靠递归状态积累先后变化。均为单5类头，另接32维统计量。三者参数约52.5/66.4/78.0万，规模小不代表缺乏拟合能力，训练分数不当测试分数。

参考[PatchTST](https://arxiv.org/abs/2211.14730)、[TSMixer](https://research.google/blog/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/)、[GRU](https://arxiv.org/abs/1406.1078)。分别借鉴共享分块、交替MLP和门控递归，不是原论文完整复现，也不把预测/翻译任务准确率当本题成绩。项目内新路线，不声称学术首创。

固定12轮AdamW4e-4、batch32、wd.01、梯度裁剪3、原类别/房间抽样。新结构交叉熵平滑.05；context_control保留原复合loss、同样12轮，故比较的是架构与配套训练方案，不是纯单因素消融。3新结构seed42三折9次、原对照2种子三折6次、新前2种子2026三折6次、e1两次、全量一次，共24次。固定末轮、不挑epoch；unknown只用于全量。

第四条独立路线参考[ROCKET](https://arxiv.org/abs/1910.13051)，固定576个随机卷积核，最大值/正响应占比及summary组成1169维特征，只拟合加权岭分类器（lambda预设10）。标准化仅训练折；固定变换可预计算所有输入，但不使用验证标签拟合。8次诊断两种子均值{random_result['candidates'][0]['two_seed_room_mean']:.2%}，明显失败，不全量训练/导出。与以上24次合计32次拟合。

|结构|种子|三房间均值|五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}

最终新候选{v} seed{s}，两种子均值{item['two_seed_room_mean']:.2%}；超过同轮数上下文对照预设指标：{sel['improved_development']}；60%达到：{sel['target_60_reached']}。合并0—4人数召回{recalls}；e1(seed42)：{item['e1']}。历史原5轮cross/context两种子均值29.24%/28.66%，训练预算不同须分开解释。开发集反复使用，不能证明统计显著或平台提升。

每布局一房间，留房间也留布局，不等于隐藏平台测试。全48记录训练实际覆盖{meta['training']['actual_windows']}/960窗口，实际48记录均参加；training_evaluation不是独立验证。

正式文件{meta['path']}，提交副本{meta['alias']}；{meta['bytes']}字节，SHA256 {meta['sha256']}。ONNX嵌入权重/接口/三个输入数值检查通过，本机4线程最慢用例平均{max(r['cpu_mean_s'] for r in resource['cases']):.4f}s、峰值提交内存{resource['peak_commit_bytes']/1e9:.3f}GB。原始输入/缓存预测差{pe:.6g}。本机资源不保证老师机器性能，平台分数未知。

权重、清单、逐例预测保存在D:/huawei2026-experiments/fresh_backbones_20261022。原所有保留模型哈希核验，0.39平台最佳不变。
'''
    (ROOT/'docs/FRESH_BACKBONES_20261022.md').write_text(report,encoding='utf-8');(WORK/'REPORT.md').write_text(report,encoding='utf-8')
    (random_root/'REPORT.md').write_text('随机字典未导出。联合报告见../fresh_backbones_20261022/REPORT.md。\n\n'+report,encoding='utf-8')
    for p in (ROOT/'experiments/random_dictionary_20261022').iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,random_root/'source_snapshot'/p.name)
    files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src').glob('*.py'))
    save(WORK/'source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    print('DELIVERED',json.dumps({k:meta[k] for k in ('architecture','seed','path','bytes','sha256','improved_development','target_60_reached')}),flush=True)
if __name__=='__main__':main()
