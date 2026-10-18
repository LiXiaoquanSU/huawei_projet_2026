import hashlib,json,os,subprocess,sys
from pathlib import Path
import blosc2,numpy as np,torch
from fusion_model import FusionModel,MEMBERS,member_path
from train_baseline import read_csv,write_csv,export_check,window
from roomsense_model import SubmissionModel
from compare_ensembles import path_for


def main():
    torch.set_num_threads(4)
    out=Path('artifacts/improved_fusion');out.mkdir(parents=True,exist_ok=True)
    room_metrics={}
    for room in ['a1','c1','g1']:
        folder=out/f'holdout_{room}';folder.mkdir(exist_ok=True)
        model=FusionModel().load_room(room)
        rows=read_csv(f'analysis/splits/holdout_{room}/val_windows.csv')
        original=[];parts=[]
        for name in MEMBERS:
            records=read_csv(path_for(name,room));mapping={(r['identifier'],r['start']):r for r in records}
            logits=np.array([[float(mapping[(r['identifier'],r['start'])][f'logit_{j}']) for j in range(5)] for r in rows])
            z=(logits-logits.mean(1,keepdims=True))/np.maximum(logits.std(1,keepdims=True),.01)
            p=np.exp(z-z.max(1,keepdims=True));parts.append(p/p.sum(1,keepdims=True))
        logits=np.log(np.maximum(np.mean(parts,axis=0),1e-8))
        provenance=dict(members=MEMBERS,room=room,seed=42,method='equal mean softmax of per-window standardized logits; output log probability',
                        checkpoint_sha256={n:hashlib.sha256(member_path(n,room).read_bytes()).hexdigest() for n in MEMBERS},
                        validation_role='adaptive development, not independent test',trained_on_other_known_rooms_only=True)
        (folder/'config.json').write_text(json.dumps(provenance,indent=2))
        torch.save(dict(model_state=model.state_dict(),config=provenance),folder/'best.pt')
        result=export_check(model,rows,folder,logits)
        room_metrics[room]=result
        if room=='a1':
            wrapper=SubmissionModel(model.eval())
            cases=[]
            for name in ['train','heldout','gap','zero']:
                with np.load(f'artifacts/model_design/case_{name}.npz') as data:
                    x,dt=data['frames'].copy(),data['intervals'].copy()
                with torch.no_grad():y=wrapper(torch.from_numpy(x),torch.from_numpy(dt)).numpy()
                np.savez(folder/f'case_{name}.npz',frames=x,intervals=dt,logits=y)
            subprocess.run([sys.executable,'-X','utf8','validate_model.py','--cpu-only','--output',str(folder),'--model-name','model.onnx'],check=True)
        print('EXPORTED',room,result['accuracy'],result['cpu_mean_s'],flush=True)
    summary=dict(members=MEMBERS,folds=room_metrics,
                 room_macro_accuracy=float(np.mean([v['accuracy'] for v in room_metrics.values()])),
                 window_accuracy=float(sum(v['accuracy']*v['samples'] for v in room_metrics.values())/sum(v['samples'] for v in room_metrics.values())),
                 parameters=sum(p.numel() for p in model.parameters()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
