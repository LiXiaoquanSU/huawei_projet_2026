"""Reuse audited preprocessing/ONNX checks; publish in a separate experiment folder."""
import shutil,importlib.util
from train_strength import *
spec=importlib.util.spec_from_file_location('strength_export_helpers',ROOT/'experiments/shared_hierarchy_20261025/export_model.py')
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
exporter.WORK=WORK;exporter.HARD=True

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.model=SharedHierarchy()
        cp=torch.load(WORK/'actual_strength_s2026/final.pt',weights_only=True);self.model.load_state_dict(cp['model_state'])
    def forward(self,t0,s0,t1,s1):
        p=self.model(torch.cat([t0,t1]),torch.cat([s0,s1])).exp().mean(0)
        return exporter.decision(p.clamp_min(1e-12).log())

def main():
    if '--publish' not in sys.argv:
        exporter.Head=Head;exporter.assemble();return
    folder=WORK/'export_hard';check=read(folder/'resource_check.json');assert check['status']=='pass'
    selection=read(WORK/'selection.json');r=selection['development'];source=read(WORK/'hard_actual_metrics.json')
    passed=r['correct']>278 and r['ci95'][0]>0 and source['correct']>=676 and all((m['correct']-m['baseline_correct'])/m['total']>=-.02 for m in r['rooms'].values())
    dest=ROOT/'models/strength_ablation_20261026';dest.mkdir(exist_ok=True);path=dest/'01_strength_hierarchy_seed2026.onnx'
    shutil.copy2(folder/'model.onnx',path)
    meta=dict(id='strength_ablation_20261026',name='绝对幅度证据分层人数模型',path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size,platform_score=None,status='unscored_candidate' if passed else 'unscored_exploratory_architecture',recommended_over054=passed,development=selection,source_metrics=source,resource_check=check,report='docs/STRENGTH_ABLATION_20261026.md')
    save(path.with_suffix('.json'),meta);save(WORK/'published.json',meta)
    reg=read(ROOT/'models/registry.json');reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('id')!=meta['id']]+[meta];reg['next_submission']=meta if passed else None;reg['research_status']='strength_ablation_completed';reg['research_report']=meta['report'];save(ROOT/'models/registry.json',reg)
    print('SAVED',path,'recommended',passed,flush=True)

if __name__=='__main__':main()
