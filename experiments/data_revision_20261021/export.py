"""Export the unchanged four-route model after data-focused dynamic pretraining."""
import gc,json,shutil,subprocess,sys
import numpy as np
import torch
from train import ROOT,WORK,read,sha,StructuredModel
from acquire import save
class Submission(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames,intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_features(t.float(),s.float())[0]
def export(variant,seed):
    key=f'{variant}_s{seed}_full';folder=WORK/'fits'/key;state=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    model=StructuredModel('spectral','physical');model.load_state_dict(state['model_state']);wrapper=Submission(model.eval()).eval()
    dest=WORK/'exports'/key;dest.mkdir(parents=True,exist_ok=True);references=[];example=None
    for path in [ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero']:
        if path=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(path) as d:x,dt=[torch.from_numpy(d[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():z=wrapper(x,dt)
        assert z.shape==(5,) and torch.isfinite(z).all();references.append(dict(input=str(path),logits=z.tolist()))
        if example is None:example=x,dt
    save(dest/'references.json',references)
    torch.onnx.export(wrapper,example,str(dest/'model.onnx'),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(dest)],check=True)
    resource=read(dest/'resource_check.json');name=('02_source_transfer' if variant=='source_transfer' else '01_'+variant)+f'_seed{seed}.onnx';target=ROOT/'models/data_revision_20261021'/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==resource['sha256']
    else:shutil.copy2(dest/'model.onnx',target)
    meta=dict(variant=variant,seed=seed,path=str(target.relative_to(ROOT)).replace('\\','/'),sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,training_records=48,parameters=state['config']['parameters'],status='research_candidate_platform_unknown',config=state['config'],resource=resource,checkpoint=str(folder/'final.pt'),topology='same original0.37 spectral/physical four-route model',data='official48 source-only pretrain control' if variant=='source_transfer' else 'EHUNAM80MHz0--4 clean multiroom counting'+(' + RadioPoints' if variant=='ehu_radio' else '')+' pretraining; official48 fine-tune',external_test='EHUNAMOffice1 entire room never trained')
    save(target.with_suffix('.json'),meta);print('EXPORTED',name,resource['bytes'],resource['sha256'],flush=True)
    del model,wrapper,state;gc.collect();return meta
def main():
    torch.set_num_threads(4);s=read(WORK/'selection.json');registry=[export(s['variant'],s['seed']),export('source_transfer',s['seed'])]
    primary=registry[0];sub=ROOT/'submissions/data_revision_20261021';sub.mkdir(parents=True,exist_ok=True)
    if (sub/'model.onnx').exists():assert sha(sub/'model.onnx')==primary['sha256']
    else:shutil.copy2(ROOT/primary['path'],sub/'model.onnx')
    save(sub/'metadata.json',primary);save(WORK/'export_registry.json',registry);save(WORK/'primary_candidate.json',primary)
    (sub/'DATA_SOURCES.md').write_text('本轮研究候选，平台分数未知。\n\nEHUNAM: de Armas et al., Scientific Data2025, DOI10.1038/s41597-025-06238-4; 数据DOI10.6084/m9.figshare.28541225，CC BY4.0。按作者元数据选80MHz0—4人PC/E，保留真实Rx1；按Subcarrier.m掩码清洗、实测时间抗混叠、边界裁剪、动态幅度归一化。外部Office1整房间留出。\n\n混合版本另用RadioPoints: https://github.com/RadioPoints/Device-free_RF_Human_Sensing_Datasets；作者README注明非商业研究并要求引用Di Domenico et al. WiMob2016/WPA2016。未复制输入天线，单Rx卷积到三Rx只做参数初始化映射。\n\n官方48条有标签数据用于最终微调。未自动上传或公开发布。\n',encoding='utf-8')
if __name__=='__main__':main()
