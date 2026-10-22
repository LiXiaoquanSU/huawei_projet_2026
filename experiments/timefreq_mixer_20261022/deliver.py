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
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models/best_037/model.onnx',ROOT/'models/best_038/model.onnx',ROOT/'models/best_039/model.onnx',ROOT/'models/best_039_context/model.onnx',ROOT/'models/ordinal_readout_20261021/01_context_ordinal_seed2026.onnx',ROOT/'models/fresh_backbones_20261022/01_mixer_seed42.onnx',ROOT/'models/spectral_diversity_20261022/01_diversity_context_seed42.onnx')}
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
    target=ROOT/'models/timefreq_mixer_20261022'/f'01_{v}_seed{s}.onnx';alias=ROOT/'submissions/timefreq_mixer_20261022/model.onnx'
    for p in (target,alias):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert sha(p)==resource['sha256']
        else:shutil.copy2(dest/'model.onnx',p)
    meta=dict(architecture=v,seed=s,path=target.relative_to(ROOT).as_posix(),alias=alias.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,status='research_candidate_platform_unverified',parameters=cp['config']['parameters'],development=item,improved_development=sel['improved_development'],target_60_reached=sel['target_60_reached'],resource=resource,training=read(folder/'metrics.json'),raw_cache_feature_error=fe,raw_cache_prediction_error=pe,checkpoint=str(folder/'final.pt'))
    for p in (target.with_suffix('.json'),alias.with_suffix('.json'),WORK/'export.json'):save(p,meta)
    fits=list((WORK/'fits').glob('*/metrics.json'));assert len(fits)==18
    for p in fits:
        m=read(p);assert m['config']['epochs']==12 and m['config']['external_data'] is False
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        if m['config']['fold']=='full':assert m['actual_records']==len({r['identifier'] for r in tr})==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in va})
            assert all(r['room'] not in ('unknown',m['config']['fold']) for r in tr)
    for p,h in protected.items():assert sha(ROOT/p)==h
    save(ROOT/'checks/timefreq_mixer_20261022.json',dict(status='pass',fits=18,record_disjoint=True,protected_models=protected,export=meta))
    reg=read(ROOT/'models/registry.json');assert reg['best_observed_platform_score']==.39
    reg['timefreq_mixer_20261022']={k:meta[k] for k in ('path','alias','sha256','bytes','platform_score','status')};reg['recommendations']['next_submission']='timefreq_mixer_20261022';reg['recommendations']['next_submission_note']='best of independent time-frequency convolution batch; compared with platform .33 Mixer; platform unverified';save(ROOT/'models/registry.json',reg)
    path=ROOT/'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['path']!=meta['path']];rows.append(dict(model=v+' seed'+str(s),platform_score='',path=meta['path'],sha256=meta['sha256'],source='time-frequency convolution research; platform unverified',status=meta['status'],submission_id=''))
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    displayed=sel['controls']+sel['screen']+[r for c in sel['candidates'] for r in c['results'] if r['seed']==2026]
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|{r['worst']:.2%}|" for r in displayed)
    cm=np.asarray([r['confusion_matrix'] for r in item['results']]).sum(0);recalls=(np.diag(cm)/cm.sum(1)).tolist()
    report=f'''# 时频卷积Mixer：小模型0.33后的独立骨干

上次MLP Mixer平台0.33已按SHA 50fc32fa…afbeaa361记录，2.76MB。小模型有一定能力，但并未超过两份0.39，不能从此推出增加参数必然有收益。本轮没有旧0.39专家/序数/存在性分支，也没有加载旧权重。

参考[ConvMixer](https://arxiv.org/abs/2201.09792)的空间/通道分离混合和[ConvNeXt](https://arxiv.org/abs/2201.03545)的深度卷积/逐位置通道MLP，改编为CSI时频结构，不是图像论文完整复现，不声称学术首创。

conv_coarse：沿用MLP Mixer的幅度128→32均值压缩，改为局部二维卷积骨干。conv_fine：将每4个细幅度频点排入内部通道，与相位32个频率区域对齐，保留预处理后的128点，不复制相位/虚构天线；只是无损重排缓存幅度，不恢复原1024子载波。axial_fine：相同fine输入，把二维核拆成频率5×1与时间1×7序列处理。三者96宽、6残差块，时间膨胀1/2/4重复，逐位置通道MLP，汇总后单分类头。输入实际时间/间隔/缺帧信息保留。局部权重共享旨在减少位置特定记忆，尚不能称房间不变。

固定12轮AdamW4e-4、batch32、wd.01、交叉熵平滑.05、梯度裁剪3、噪声.02/频率drop5%、原类别/房间抽样。协议匹配已测MLP Mixer。三方案seed42三折9次，前二seed2026三折6次、e1两次、全量一次，共18次。复用Mixer两种子对照，并复算seed42 a1预测一致。固定末轮不挑epoch，按min(三房间均值,五类宏召回)排序。

|结构|种子|三房间均值|五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}

新候选{v} seed{s}，两种子三房间均值{item['two_seed_room_mean']:.2%}；超过原MLP Mixer的预设指标：{sel['improved_development']}；达到60%：{sel['target_60_reached']}。合并0—4人数召回{recalls}；e1(seed42)：{item['e1']}。历史原5轮cross/context两种子均值29.24%/28.66%，平台均0.39；训练预算不同，分开解释。开发集反复使用，不能证明统计显著或平台改善。

数据只有每布局一个房间，留房间同时留布局。官方缓存960窗口，unknown不进开发，全量实际48记录、{meta['training']['actual_windows']}/960不同窗口；训练数据重评不是独立测试。未用外部或U数据。

正式文件{meta['path']}，提交副本{meta['alias']}。{meta['bytes']}字节，SHA256 {meta['sha256']}。ONNX嵌入权重/接口/三输入数值检查通过，本机4线程最慢用例平均{max(r['cpu_mean_s'] for r in resource['cases']):.4f}s，峰值提交内存{resource['peak_commit_bytes']/1e9:.3f}GB；不保证老师机器速度。原始输入/缓存预测差{pe:.6g}。平台成绩未知，原最佳0.39保持。

全部权重、清单与逐例预测在D:/huawei2026-experiments/timefreq_mixer_20261022。所有旧保留模型受哈希核验保护。
'''
    (ROOT/'docs/TIMEFREQ_MIXER_20261022.md').write_text(report,encoding='utf-8');(WORK/'REPORT.md').write_text(report,encoding='utf-8')
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    print('DELIVERED',json.dumps({k:meta[k] for k in ('architecture','seed','path','bytes','sha256','improved_development','target_60_reached')}),flush=True)
if __name__=='__main__':main()
