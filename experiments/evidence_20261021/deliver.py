"""Export selected development improvement without promoting its platform score."""
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
    torch.set_num_threads(4)
    tag=WORK.name
    selection=read(WORK/'selection.json');assert selection['improved_development']
    protected={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'models/best_037/model.onnx',ROOT/'models/best_038/model.onnx',ROOT/'models/best_039/model.onnx',ROOT/'submissions/global_local_20261021/alternative_model.onnx']}
    item=selection['candidates'][0];v,s=item['architecture'],item['seed']
    folder=WORK/'fits'/f'{v}_s{s}_full';cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    model=build(v).eval();model.load_state_dict(cp['model_state']);wrapper=Submission(model).eval()
    dest=WORK/'exports'/f'{v}_s{s}';dest.mkdir(parents=True,exist_ok=True);references=[]
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
    resource=read(dest/'resource_check.json');d=load();row=d['rows'][0];raw=Path('D:/huawei2026-data/labeled');a,b=int(row['start']),int(row['stop'])
    c=blosc2.load_array(str(raw/f"CSI_{row['identifier']}.b2"));ati=blosc2.load_array(str(raw/f"ATI_{row['identifier']}.b2"))
    x=torch.from_numpy(c[a:b].view(np.float32).reshape(600,1,3,1024,2).copy());dt=torch.from_numpy(ati[a:b].copy());dt[-1]=0
    with torch.no_grad():
        t,sm=model.preprocess(x[None].double(),dt[None].double());t,sm=t.float(),sm.float()
        fe=float((t-d['t'][:1]).abs().max());pe=float((model.forward_features(t,sm)-model.forward_features(d['t'][:1],d['s'][:1])).abs().max())
    assert fe<.001 and pe<.005,(fe,pe)
    target=ROOT/'models'/tag/f'01_{v}_seed{s}.onnx';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==resource['sha256']
    else:shutil.copy2(dest/'model.onnx',target)
    alias=ROOT/'submissions'/tag/'model.onnx';alias.parent.mkdir(parents=True,exist_ok=True)
    if alias.exists():assert sha(alias)==resource['sha256']
    else:shutil.copy2(target,alias)
    meta=dict(architecture=v,seed=s,path=target.relative_to(ROOT).as_posix(),alias=alias.relative_to(ROOT).as_posix(),bytes=resource['bytes'],sha256=resource['sha256'],parameters=cp['config']['parameters'],platform_score=None,status='development_improvement_platform_unverified',development=item,resource=resource,checkpoint=str(folder/'final.pt'),raw_cache_feature_max_error=fe,raw_cache_prediction_max_error=pe,training=read(folder/'metrics.json'))
    save(target.with_suffix('.json'),meta);save(alias.with_suffix('.json'),meta);save(WORK/'export.json',meta)
    fits=[]
    for p in sorted((WORK/'fits').glob('*/metrics.json')):
        m=read(p);fits.append(m)
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
        if m['config']['fold']=='full':assert len({r['identifier'] for r in tr})==m['actual_records']==48
        else:
            with (p.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as f:va=list(csv.DictReader(f))
            assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in va})
            assert all(r['room'] not in ('unknown',m['config']['fold']) for r in tr)
    for p,h in protected.items():assert sha(ROOT/p)==h
    audit=dict(status='pass',fits=len(fits),protected=protected,record_disjoint=True,export=meta)
    save(WORK/'audit.json',audit);save(ROOT/'checks'/f'{tag}.json',audit)
    registry=read(ROOT/'models/registry.json');assert registry['best_observed_platform_score']==.39
    registry[tag]=dict(platform_score=None,status=meta['status'],primary=meta['path'],alias=meta['alias'],sha256=meta['sha256'],report=f'docs/{tag.upper()}.md');save(ROOT/'models/registry.json',registry)
    scorepath=ROOT/'models/platform_scores.csv'
    with scorepath.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['path']!=meta['path']]
    rows.append(dict(model=v+' seed'+str(s),platform_score='',path=meta['path'],sha256=meta['sha256'],source='matched development only',status=meta['status'],submission_id=''))
    with scorepath.open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    rows=selection['controls']+selection['screen']+[r for candidate in selection['candidates'] for r in candidate['results'] if r['seed']==2026]
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|{r['worst']:.2%}|" for r in rows)
    controls=selection['controls'];base=float(np.mean([r['room_mean'] for r in controls if r['architecture']=='cross_events']));new=item['two_seed_room_mean']
    cm=np.asarray([r['confusion_matrix'] for r in item['results']]).sum(0);recalls=(np.diag(cm)/cm.sum(1)).tolist()
    report=f'''# 0.39之后的对应证据与实际时间结构实验

平台最佳仍0.39；本候选平台未测。本地两种子三房间均值{base:.2%}→{new:.2%}，不是平台成绩预测。完整完成{len(fits)}次新拟合，包含3次原0.39对照复现；其seed42合并混淆矩阵与历史结果一致。控制seed2026为历史同协议结果。

结构定义与预设选择见experiments/evidence_20261021/PLAN.md。aligned是同天线对/频率/时间片段幅相联合证据，timed是实际时间距离注意力；研究阶段未限制参数量，不使用外部数据或U。timed仍使用150点缓存，不宣称已恢复600帧细节。所有拟合从随机初始化，5轮原AdamW/loss/采样，按完整房间划分，无全量权重泄漏。e1只作压力诊断。

|结构|种子|三房间均值|五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}

选中{v} seed{s}，全量训练实际覆盖48条有标签记录，实际不同窗口数{meta['training']['actual_windows']}/960。原始记录稀少且房间/人数耦合；多次使用的本地开发集不是盲测，两种子不足以证明统计显著。候选两种子各人数召回：{recalls}。seed42 e1诊断：{item['e1']}。

正式模型：{meta['path']}；提交副本：{meta['alias']}。SHA256 {meta['sha256']}，{meta['bytes']}字节。真实/断包/全零输入ONNX数值及CPU资源检查通过；最慢用例平均{max(c['cpu_mean_s'] for c in resource['cases']):.4f}秒，峰值提交内存{resource['peak_commit_bytes']/1e9:.3f}GB。本机测试不保证平台机器耗时。原始输入/缓存特征误差{fe:.6g}，预测误差{pe:.6g}。旧0.37/0.38/0.39与待测上下文候选哈希未变。

不能保证0.5；提交后只按对应SHA登记新成绩，未测试前不替换平台最佳。完整记录D:/huawei2026-experiments/evidence_20261021；审计checks/evidence_20261021.json。
'''
    if tag=='query_20261021':
        report=report.replace('包含3次原0.39对照复现；其seed42合并混淆矩阵与历史结果一致。控制seed2026为历史同协议结果。','两种0.39对照均复用历史同协议结果；跨注意力seed42在本轮第一阶段已复现。')
        report=report.replace('aligned是同天线对/频率/时间片段幅相联合证据，timed是实际时间距离注意力','query_context是五个人数假设分别查询四路特征和五段上下文事件，兼容分数加入原分类logit')
        report=report.replace('timed仍使用150点缓存，不宣称已恢复600帧细节。','仍使用原150点缓存。')
        report=report.replace('evidence_20261021；审计checks/evidence_20261021.json','query_20261021；审计checks/query_20261021.json')
    (ROOT/'docs'/f'{tag.upper()}.md').write_text(report,encoding='utf-8');(WORK/'REPORT.md').write_text(report,encoding='utf-8')
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    print('DELIVERED',json.dumps(dict(path=str(alias),sha256=meta['sha256'],bytes=meta['bytes'],local_mean=new,baseline=base)),flush=True)
if __name__=='__main__':main()
