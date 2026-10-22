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
    if (WORK/'platform_feedback_037.json').exists():
        raise RuntimeError('Candidate already evaluated at 0.37; do not overwrite feedback with an untested export.')
    torch.set_num_threads(4);sel=read(WORK/'selection.json');item=sel['candidates'][0];v,s=item['architecture'],item['seed']
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models/best_037/model.onnx',ROOT/'models/best_038/model.onnx',ROOT/'models/best_039/model.onnx',ROOT/'models/best_039_context/model.onnx',ROOT/'models/ordinal_readout_20261021/01_context_ordinal_seed2026.onnx')}
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
    target=ROOT/'models/spatial_research_20261021'/f'01_{v}_seed{s}.onnx';alias=ROOT/'submissions/spatial_research_20261021/model.onnx'
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
    save(ROOT/'checks/spatial_research_20261021.json',dict(status='pass',fits=18,record_disjoint=True,protected_models=protected,export=meta))
    reg=read(ROOT/'models/registry.json');assert reg['best_observed_platform_score']==.39
    reg['spatial_research_20261021']={k:meta[k] for k in ('path','alias','sha256','bytes','platform_score','status')};reg['recommendations']['next_submission']='spatial_research_20261021';reg['recommendations']['next_submission_note']='best of this spatial research batch; platform unverified';save(ROOT/'models/registry.json',reg)
    path=ROOT/'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['path']!=meta['path']];rows.append(dict(model=v+' seed'+str(s),platform_score='',path=meta['path'],sha256=meta['sha256'],source='spatial architecture research; platform unverified',status=meta['status'],submission_id=''))
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    displayed=sel['controls']+sel['screen']+[r for c in sel['candidates'] for r in c['results'] if r['seed']==2026]
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|{r['worst']:.2%}|" for r in displayed)
    cm=np.asarray([r['confusion_matrix'] for r in item['results']]).sum(0);recalls=(np.diag(cm)/cm.sum(1)).tolist()
    report=f'''# 空间结构研究与下一候选

用户反馈上轮序数读出0.38，低于原两个0.39，已撤回推荐并保留记录。其本地改善未迁移到平台，不能继续只按开发峰值修改输出。本轮所有方案保留原分类/序数平均输出。

查阅[2025 CSI元学习](https://arxiv.org/abs/2502.03117)、[2026 CSI-ResNet-A预印本](https://arxiv.org/abs/2601.02203)、[AFF](https://arxiv.org/abs/2009.14082)、[Res2Net](https://arxiv.org/abs/1904.01169)。前两篇的目标域有标签适配/进出事件条件不直接匹配本题；具体网络、借鉴边界和预设计划见experiments/spatial_research_20261021/PLAN.md。不是论文完整复现，不把活动或进出计数分数当本题精确人数分数。

aff_context：原残差与shortcut通过局部/全局上下文门控融合，初始化预测等价于原模型。res2_context：块内分组逐级卷积建立多尺度。pyramid_context：第二/三层有序频区统计通过门控补充末层表示。均保留context_events时间路线及原训练/loss。无外部/U/全量旧权重预训练。研究阶段不按100MB截断想法。

固定五轮AdamW4e-4、batch32、weight decay.01、梯度裁剪3、原类别/房间抽样；3结构seed42筛选9次、前2结构seed2026复验6次、e1两次、全量一次，共18次。原对照复用，context seed2026 a1预测复算一致。e1非唯一淘汰门槛，固定末轮不挑epoch。

|结构|种子|三房间均值|五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}

本轮选择{v} seed{s}，两种子三房间均值{item['two_seed_room_mean']:.2%}，预设选择指标超过两种原0.39对照：{sel['improved_development']}。60%目标达到：{sel['target_60_reached']}。两种子合并各人数召回：{recalls}；seed42 e1：{item['e1']}。训练集分数不是测试分数，本地开发被反复使用，两个种子不证明统计显著或平台改善。

有标签数据每种已知布局仅一个物理房间（a→a1、b→b1、c→c1、e→e1、g→g1），因此本地留房间同时留布局，比同布局新房间更严格；它与平台测试条件不完全相同。保留既定划分，不用随机窗口或同房间测试凑60%。缺少房间与人数的交叉组合仍是识别泛化能力的主要限制。

最终全48记录训练，实际不同窗口覆盖{meta['training']['actual_windows']}/960。提交文件{meta['alias']}，正式文件{meta['path']}；{meta['bytes']}字节，SHA256 {meta['sha256']}，平台未知。三种输入数值检查/CPU资源通过：最慢用例平均{max(r['cpu_mean_s'] for r in resource['cases']):.4f}s，峰值提交内存{resource['peak_commit_bytes']/1e9:.3f}GB；不保证平台机器速度。原始输入/缓存预测误差{pe:.6g}。

原0.37/0.38/两份0.39/本次已测序数0.38均哈希保护。全部权重、逐例预测和清单在D:/huawei2026-experiments/spatial_research_20261021。平台最佳仍0.39；本文件是下一研究候选，不能称已达到0.6或超过0.39。
'''
    (ROOT/'docs/SPATIAL_RESEARCH_20261021.md').write_text(report,encoding='utf-8');(WORK/'REPORT.md').write_text(report,encoding='utf-8')
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    print('DELIVERED',json.dumps({k:meta[k] for k in ('architecture','seed','path','bytes','sha256','improved_development','target_60_reached')}),flush=True)
if __name__=='__main__':main()
