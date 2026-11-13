"""Held-record dependency probes. Incoherent interventions are sensitivity, not accuracy claims."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/mechanism_20261029');CACHE=Path('D:/huawei2026-experiments/next053_20261025')

def main():
    WORK.mkdir(exist_ok=True);torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    t=torch.from_numpy(np.array(np.load(CACHE/'t.npy',mmap_mode='r'))).cuda();s=torch.from_numpy(np.array(np.load(CACHE/'s.npy',mmap_mode='r'))).cuda()
    ix,sg,si=[v.cuda() for v in old.permutation_indices((1,0,2))];results=[]
    for architecture in ('stable_chain_20261028','texture_chain_20261028'):
        spec=importlib.util.spec_from_file_location(architecture,ROOT/'experiments'/architecture/'network.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for fold in ('record0','record1','record2'):
          folder=Path('D:/huawei2026-experiments')/architecture/f'{fold}_full_s42';d=torch.load(folder/'predictions.pt',weights_only=True)
          for checkpoint in ('final','last'):
            cp=torch.load(folder/f'{checkpoint}.pt',weights_only=True);m=mod.EvidenceChain().cuda().eval();m.load_state_dict(cp['model_state']);ids=d['test_indices'];y=d['labels'][ids];ps={}
            for condition in ('original','coherent_rx','phase_assignment_only','phase_time_shift','static_other_window'):
                out=[]
                with torch.inference_mode():
                    for start in range(0,len(ids),24):
                        jj=ids[start:start+24];a=t[jj].clone();b=s[jj].clone()
                        if condition=='coherent_rx':a=a.index_select(2,ix)*sg[None];b=b.index_select(2,si)
                        elif condition=='phase_assignment_only':a[:,:,768:960]=a.index_select(2,ix)[:,:,768:960]*sg[None,:,768:960]
                        elif condition=='phase_time_shift':
                            a[:,0,768:960]=a[:,0,768:960].roll(30,-1);a[:,1,768:960]=a[:,1,768:960].roll(-30,-1)
                        elif condition=='static_other_window':
                            others=[]
                            for j in jj:
                                same=[k for k in ids if d['rows'][k]['identifier']==d['rows'][j]['identifier']];others.append(same[(same.index(j)+1)%len(same)])
                            a[:,:,:384]=t[others,:,:384]
                        p=m(a.flatten(0,1),b.flatten(0,1)).exp().reshape(-1,2,5).mean(1);out.append(p.cpu())
                ps[condition]=torch.cat(out)
            base=ps['original'];reference=d['logits' if checkpoint=='final' else 'last_logits'][ids].exp();error=float((base-reference).abs().max());assert error<1e-4,(architecture,fold,checkpoint,error)
            assert float((base-ps['coherent_rx']).abs().max())<1e-4
            for condition,p in ps.items():
                results.append(dict(architecture=architecture,fold=fold,checkpoint=checkpoint,epoch=cp['epoch'],condition=condition,total=len(y),correct=int((p.argmax(-1)==y).sum()),changed_predictions=int((p.argmax(-1)!=base.argmax(-1)).sum()),mean_probability_L1=float((p-base).abs().sum(-1).mean()),max_probability_error=float((p-base).abs().max())))
            print('AUDIT',architecture,fold,checkpoint,flush=True);del m;torch.cuda.empty_cache()
    old.save(WORK/'audit.json',dict(results=results,interpretation='Coherent permutation is mathematical invariance. Phase assignment and phase time shift can be physically inconsistent/OOD; sensitivity is not evidence of causal physical understanding. Static substitution preserves recording label but not exact joint signal. No intervention is used to select a submission.',code_sha256=old.sha(Path(__file__))))

if __name__=='__main__':main()
