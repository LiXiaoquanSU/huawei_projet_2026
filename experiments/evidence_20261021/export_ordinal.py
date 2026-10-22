"""Export existing context weights with the fixed ordinal readout ablation."""
import csv,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from train import ROOT,OLD,read,save,sha,load
from network import old

WORK=Path('D:/huawei2026-experiments/ordinal_readout_20261021')
class OrdinalSubmission(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames,intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_tasks(t.float(),s.float())['ordinal'].clamp_min(1e-8).log()[0]

def main():
    if (WORK/'platform_feedback_038.json').exists():raise RuntimeError('This model now has platform feedback .38; use a new experiment directory.')
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    source=ROOT/'models/global_local_20261021/02_context_events_seed2026.onnx';expected='f2ed6fb013e74cdf8d22f3597d11250df517f73a852048603efa64993b50cfb8';assert sha(source)==expected
    cp_path=OLD/'fits/context_events_s2026_full/final.pt';cp=torch.load(cp_path,map_location='cpu',weights_only=True)
    model=old.build('context_events').eval();model.load_state_dict(cp['model_state']);wrapper=OrdinalSubmission(model).eval();references=[]
    for path in [ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero']:
        if path=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(path) as data:x,dt=[torch.from_numpy(data[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():out=wrapper(x,dt)
        assert out.shape==(5,) and torch.isfinite(out).all();references.append(dict(input=str(path),logits=out.tolist()))
        if len(references)==1:example=x,dt
    save(WORK/'references.json',references)
    torch.onnx.export(wrapper,example,str(WORK/'model.onnx'),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(WORK)],check=True)
    resource=read(WORK/'resource_check.json');diagnosis=read(ROOT/'checks/head_readout_diagnosis.json')
    rr=[r for r in diagnosis['results'] if r['architecture']=='context_events' and r['readout']=='ordinal'];mean=float(np.mean([r['room_mean'] for r in rr]));macro=float(np.mean([r['macro_recall'] for r in rr]))
    target=ROOT/'models/ordinal_readout_20261021/01_context_ordinal_seed2026.onnx';target.parent.mkdir(parents=True,exist_ok=True)
    alias=ROOT/'submissions/ordinal_readout_20261021/model.onnx';alias.parent.mkdir(parents=True,exist_ok=True)
    for p in (target,alias):
        if p.exists():assert sha(p)==resource['sha256']
        else:shutil.copy2(WORK/'model.onnx',p)
    # Diagnostic e1 checkpoint was trained with seed42; not the final seed2026.
    d=load();stress_cp=torch.load(OLD/'fits/context_events_s42_e1/final.pt',map_location='cpu',weights_only=True);model.load_state_dict(stress_cp['model_state']);ix=[i for i,r in enumerate(d['rows']) if r['room']=='e1'];cm=torch.zeros(5,5,dtype=torch.long)
    with torch.no_grad():
        for start in range(0,len(ix),8):
            j=ix[start:start+8];pred=model.forward_tasks(d['t'][j],d['s'][j])['ordinal'].argmax(1)
            for y,p in zip(d['y'][j],pred):cm[y,p]+=1
    meta=dict(architecture='context_events_ordinal_readout',seed=2026,path=target.relative_to(ROOT).as_posix(),alias=alias.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,status='fixed_readout_candidate_platform_unverified',source_platform_model_sha256=expected,source_checkpoint=str(cp_path),source_checkpoint_sha256=sha(cp_path),new_training_fits=0,training=cp['config'],development=dict(two_seed_room_mean=mean,two_seed_macro_recall=macro,results=rr),e1_seed42=dict(accuracy=float(cm.diag().sum()/cm.sum()),confusion_matrix=cm.tolist()),resource=resource,note='Reuse all48-record trained context seed2026 weights; remove final direct/ordinal averaging. No mixing coefficient search; not yet platform evaluated.')
    for p in (target.with_suffix('.json'),alias.with_suffix('.json'),WORK/'export.json'):save(p,meta)
    assert sha(source)==expected
    reg=read(ROOT/'models/registry.json');assert reg['best_observed_platform_score']==.39
    reg['ordinal_readout_20261021']={k:meta[k] for k in ('path','alias','sha256','bytes','platform_score','status')};reg['ordinal_readout_20261021']['report']='docs/ORDINAL_READOUT_20261021.md';save(ROOT/'models/registry.json',reg)
    scorepath=ROOT/'models/platform_scores.csv'
    with scorepath.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['path']!=meta['path']];rows.append(dict(model='context ordinal readout seed2026',platform_score='',path=meta['path'],sha256=meta['sha256'],source='fixed readout ablation; no platform evaluation',status=meta['status'],submission_id=''))
    with scorepath.open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    table='\n'.join(f"|{r['architecture']}|{r['seed']}|{r['readout']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|" for r in diagnosis['results'])
    report=f'''# 固定序数读出：上下文事件候选

两个已评测模型仍并列平台0.39；本候选平台未知。复用context_events seed2026在全部48记录上训练的现有权重，没有新训练。唯一推理变化是从原分类/序数概率各半改为只输出已有CORN式序数概率；不是新增序数头，不删除训练时的分类监督。ONNX可裁掉不再参与输出的分支。

在查看诊断前固定direct/ordinal/原平均三种对照，无混合系数搜索。使用每折仅源数据训练的检查点评估，未把全量模型带入留出房间。两种结构×两种子共四组中序数读出房间均值均提高；上下文两种子均值28.66%→{mean:.2%}，跨注意力29.24%→30.85%。本地开发反复使用，不能保证平台提高。全部读出本地4人召回仍为0。

|结构|种子|读出|三房间均值|五类宏召回|
|---|---:|---|---:|---:|
{table}

序数头使用四个条件概率的累积乘积构造0—4类分布，最后仍argmax分类；并不是把连续人数四舍五入。现有任务主CE训练direct，另有序数BCE及一致性项，因此direct和ordinal可能发生预测冲突；本轮结果支持测试不同读出，不证明序数在所有域更好。

最终采用已有平台0.39上下文模型seed2026，不按这次seed42更高的单项均值重新挑全量seed。e1压力诊断使用独立seed42折模型：{meta['e1_seed42']}。不能把它当最终seed2026成绩。

提交：{meta['alias']}；正式文件：{meta['path']}。大小{meta['bytes']/1e6:.2f}MB，SHA256 {meta['sha256']}。本机真实/断包/全零ONNX数值、内嵌权重、接口和资源检查通过，最慢用例平均{max(r['cpu_mean_s'] for r in resource['cases']):.4f}s，峰值提交内存{resource['peak_commit_bytes']/1e9:.3f}GB。原0.39文件哈希保持不变。源检查点SHA {meta['source_checkpoint_sha256']}。
'''
    (ROOT/'docs/ORDINAL_READOUT_20261021.md').write_text(report,encoding='utf-8');(WORK/'REPORT.md').write_text(report,encoding='utf-8');save(ROOT/'checks/ordinal_readout_20261021.json',meta)
    print('ORDINAL_EXPORTED',json.dumps(dict(path=str(alias),bytes=meta['bytes'],sha256=meta['sha256'],local_mean=mean)),flush=True)
if __name__=='__main__':main()
