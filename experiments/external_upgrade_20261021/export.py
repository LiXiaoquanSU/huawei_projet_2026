"""Export full-data research candidates and check them in isolated CPU processes."""
import gc,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper,helper,TensorProto
import torch
from net import ROOT,fresh,Submission,StructuredModel
from prepare import WORK,save
from run import read,sha

def export(variant,seed):
    key=f'{variant}_s{seed}_full';folder=WORK/'fits'/key
    state=torch.load(folder/'final.pt',weights_only=True,map_location='cpu')
    model=fresh(seed);model.load_state_dict(state['model_state']);wrapper=Submission(model.eval()).eval()
    out=WORK/'exports'/key;out.mkdir(parents=True,exist_ok=True)
    cases=[ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero'];references=[];example=None
    for path in cases:
        if path=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(path) as d:x,dt=[torch.from_numpy(d[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():z=wrapper(x,dt)
        assert z.shape==(5,) and torch.isfinite(z).all();references.append(dict(input=str(path),logits=z.tolist()))
        if example is None:example=(x,dt)
    save(out/'references.json',references)
    torch.onnx.export(wrapper,example,str(out/'model.onnx'),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(out)],check=True)
    resource=read(out/'resource_check.json')
    name=('02_source_control' if variant=='source_control' else '01_'+variant)+f'_seed{seed}.onnx'
    target=ROOT/'models/external_upgrade_20261021'/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==resource['sha256']
    else:shutil.copy2(out/'model.onnx',target)
    meta=dict(variant=variant,seed=seed,path=str(target.relative_to(ROOT)).replace('\\','/'),bytes=resource['bytes'],sha256=resource['sha256'],platform_score=None,training_records=48,checkpoint=str(folder/'final.pt'),resource=resource,config=state['config'],status='research_candidate_platform_unknown',source='official48 only' if variant=='source_control' else 'official48 + RadioPoints0--4 + Brescia0--4 external_train'+(' + SHARP activity/presence' if variant!='external_counts' else ''),note='Full-data training accuracy is not an independent evaluation. Existing platform 0.37 models remain best confirmed.')
    save(target.with_suffix('.json'),meta)
    print('EXPORTED',json.dumps(meta,ensure_ascii=False),flush=True)
    del state,model,wrapper;gc.collect();return meta
class Ensemble(torch.nn.Module):
    def __init__(self,models):super().__init__();self.models=torch.nn.ModuleList(models)
    def forward(self,frames,intervals):
        t,s=self.models[0].preprocess(frames[None].double(),intervals[None].double());t,s=t.float(),s.float()
        p=(self.models[0].forward_features(t,s).exp()+self.models[1].forward_features(t,s).exp())*.5
        return p.clamp_min(1e-8).log()[0]
def export_ensemble(variant,kind='external_pair',selected_seed=42):
    models=[]
    if kind=='anchor_pair':
        state=torch.load('D:/huawei2026-experiments/random_restart_20261020/fits/original_s42_full/final.pt',map_location='cpu',weights_only=True)
        model=StructuredModel('spectral','physical');model.load_state_dict(state['model_state']);models.append(model.eval());del state
    for seed in ((selected_seed,) if kind=='anchor_pair' else (42,2026)):
        state=torch.load(WORK/'fits'/f'{variant}_s{seed}_full/final.pt',map_location='cpu',weights_only=True)
        m=fresh(seed);m.load_state_dict(state['model_state']);models.append(m.eval());del state
    wrapper=Ensemble(models).eval();out=WORK/'exports'/f'{variant}_{kind}';out.mkdir(parents=True,exist_ok=True)
    references=[];example=None
    for path in [ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero']:
        if path=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(path) as d:x,dt=[torch.from_numpy(d[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():z=wrapper(x,dt)
        assert torch.isfinite(z).all();references.append(dict(input=str(path),logits=z.tolist()))
        if example is None:example=x,dt
    save(out/'references.json',references)
    full=out/'fp32_reference.onnx'
    torch.onnx.export(wrapper,example,str(full),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    graph=onnx.load(full);cast_nodes=[];converted=0
    for tensor in graph.graph.initializer:
        if tensor.data_type!=TensorProto.FLOAT:continue
        a=numpy_helper.to_array(tensor)
        if a.size<256:continue
        name=tensor.name;stored=name+'__fp16_storage'
        tensor.CopyFrom(numpy_helper.from_array(a.astype(np.float16),name=stored))
        cast_nodes.append(helper.make_node('Cast',[stored],[name],to=TensorProto.FLOAT,name=name+'__restore_fp32'));converted+=1
    original_nodes=list(graph.graph.node);del graph.graph.node[:];graph.graph.node.extend(cast_nodes+original_nodes)
    onnx.checker.check_model(graph);onnx.save(graph,out/'model.onnx');del graph
    del wrapper,models;gc.collect()
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(out)],check=True)
    resource=read(out/'resource_check.json');target=ROOT/'models/external_upgrade_20261021'/f'00_{kind}.onnx'
    if target.exists():assert sha(target)==resource['sha256']
    else:shutil.copy2(out/'model.onnx',target)
    meta=dict(variant=variant,seeds=[42,selected_seed] if kind=='anchor_pair' else [42,2026],path=str(target.relative_to(ROOT)).replace('\\','/'),bytes=resource['bytes'],sha256=resource['sha256'],platform_score=None,training_records=48,status='research_candidate_platform_unknown',resource=resource,storage='FP16 initializers restored to FP32 by Cast; FP32 compute',converted_tensors=converted,ensemble=kind+'; equal probability mean; shared preprocessing',note='Local room-disjoint development passed the predeclared ensemble threshold; platform score unknown.')
    save(target.with_suffix('.json'),meta);return meta
def main():
    torch.set_num_threads(4);s=read(WORK/'selection.json')
    results=[export(s['variant'],s['seed']),export('source_control',s['seed'])]
    primary=results[0]
    if s.get('ensemble_kind','single')!='single':
        try:
            meta=export_ensemble(s['variant'],s['ensemble_kind'],s['seed']);results.append(meta);primary=meta
        except (subprocess.CalledProcessError,AssertionError) as error:
            save(WORK/'ensemble_export_failure.json',dict(error=str(error),fallback=primary['path']));print('ENSEMBLE_RESOURCE_OR_NUMERICAL_CHECK_FAILED_USING_SINGLE',str(error),flush=True)
    sub=ROOT/'submissions/external_upgrade_20261021';sub.mkdir(parents=True,exist_ok=True)
    if (sub/'model.onnx').exists():assert sha(sub/'model.onnx')==primary['sha256']
    else:shutil.copy2(ROOT/primary['path'],sub/'model.onnx')
    save(sub/'metadata.json',primary);save(WORK/'primary_candidate.json',primary)
    (sub/'DATA_SOURCES.md').write_text('研究候选，平台成绩未知。\n\nRadioPoints: https://github.com/RadioPoints/Device-free_RF_Human_Sensing_Datasets — 原README注明非商业研究并要求引用Di Domenico et al. WiMob2016/WPA2016。\nBrescia: https://zenodo.org/records/17225694 — CC BY-SA4.0，变换与原始归属见研究报告。\nSHARP（多任务版本）：https://researchdata.cab.unipd.it/624/ — 数据页ODC Attribution；作者代码GPL单独保留。\n\n本地训练和导出已完成，不代表已核定全部比赛使用/再分发条款；文件未自动上传或公开发布。\n',encoding='utf-8')
    save(WORK/'export_registry.json',results)
if __name__=='__main__':main()
