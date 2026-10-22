"""Fixed readout ablation: direct vs ordinal vs existing equal average."""
import gc
import numpy as np
import torch
from train import ROOT,OLD,load,save
from network import old
def main():
    torch.set_num_threads(4);d=load();results=[]
    for v in ('cross_events','context_events'):
        for s in (42,2026):
            metrics={h:[] for h in ('direct','ordinal','average')}
            for f in ('a1','c1','g1'):
                cp=torch.load(OLD/'fits'/f'{v}_s{s}_{f}/final.pt',map_location='cpu',weights_only=True)
                model=old.build(v).cuda().eval();model.load_state_dict(cp['model_state']);ix=[i for i,r in enumerate(d['rows']) if r['room']==f]
                cms={h:np.zeros((5,5),int) for h in metrics}
                with torch.no_grad():
                    for start in range(0,len(ix),32):
                        j=ix[start:start+32];out=model.forward_tasks(d['t'][j].cuda(),d['s'][j].cuda())
                        for h,p in [('direct',out['direct']),('ordinal',out['ordinal']),('average',out['logits'])]:
                            for y,pred in zip(d['y'][j].tolist(),p.argmax(1).cpu().tolist()):cms[h][y,pred]+=1
                for h,cm in cms.items():metrics[h].append(dict(room=f,accuracy=float(np.trace(cm)/cm.sum()),confusion_matrix=cm.tolist()))
                del model,cp;gc.collect();torch.cuda.empty_cache()
            for h,rr in metrics.items():
                cm=np.sum([r['confusion_matrix'] for r in rr],0);mean=float(np.mean([r['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
                results.append(dict(architecture=v,seed=s,readout=h,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),confusion_matrix=cm.tolist(),rooms=rr))
            print(v,s,{h:float(np.mean([r['accuracy'] for r in rr])) for h,rr in metrics.items()},flush=True)
    save(ROOT/'checks/head_readout_diagnosis.json',dict(results=results,note='fixed discrete readout ablation; no weights tuned; fold checkpoints exclude validation rooms'))
if __name__=='__main__':main()
