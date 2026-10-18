"""Refit frozen fusion on all labels. No local held-out accuracy is claimed."""
import hashlib,json,random,subprocess,sys
from pathlib import Path
from time import perf_counter
import blosc2,numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from fusion_model import FusionModel,MEMBERS
from improved_models import ImprovedModel
from tf_model import TFModel
from roomsense_model import SubmissionModel,export_submission
from run_iterations import load_bank
from train_baseline import read_csv,write_csv,window


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=Path('artifacts/submission_candidate')
    if (out/'final.pt').exists():raise FileExistsError('Candidate already trained; preserve it')
    out.mkdir(parents=True,exist_ok=True)
    index=read_csv('D:/huawei2026-data/labeled/samples.csv')
    unknown=[r for r in index if r['room']=='unknown']
    unknown_rows=[]
    for row in unknown:
        for start in range(0,12000,600):
            unknown_rows.append(dict(row,start=str(start),stop=str(start+600),
                           csi_path=f'D:/huawei2026-data/labeled/CSI_{row["identifier"]}.b2',
                           ati_path=f'D:/huawei2026-data/labeled/ATI_{row["identifier"]}.b2'))
    fusion=FusionModel();histories={};configs=[]
    for name in MEMBERS:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=(ImprovedModel('whiten') if name=='whiten' else TFModel(name)).cuda()
        bank=load_bank(tf=name!='whiten');t,s,y,known_rows=bank
        extra_t=[];extra_s=[]
        with torch.no_grad():
            for row in unknown:
                entries=[r for r in unknown_rows if r['identifier']==row['identifier']]
                csi=blosc2.load_array(entries[0]['csi_path']);ati=blosc2.load_array(entries[0]['ati_path'])
                for start in range(0,20,4):
                    arrays=[window(csi,ati,r) for r in entries[start:start+4]]
                    x=torch.from_numpy(np.stack([a[0] for a in arrays])).cuda();dt=torch.from_numpy(np.stack([a[1] for a in arrays])).cuda()
                    f=model.preprocess(x,dt)
                    if name=='whiten':a,b=f
                    else:a,b=f,torch.zeros(len(x),17,device='cuda')
                    extra_t.append(a.cpu());extra_s.append(b.cpu())
                del csi,ati
        t=torch.cat([t,*extra_t]);s=torch.cat([s,*extra_s]);y=torch.cat([y,torch.tensor([int(r['headcount']) for r in unknown_rows])])
        rows=known_rows+unknown_rows
        assert len(rows)==960 and len({r['identifier'] for r in rows})==48
        groups={}
        def group(r):return r['room'] if r['room']!='unknown' else 'unknown_trace_'+r['identifier']
        for r in rows:groups[(int(r['headcount']),group(r))]=groups.get((int(r['headcount']),group(r)),0)+1
        room_counts={c:len({g for a,g in groups if a==c}) for c in range(5)}
        weights=[1/(groups[(int(r['headcount']),group(r))]*room_counts[int(r['headcount'])]) for r in rows]
        sampler=WeightedRandomSampler(weights,len(rows),replacement=True,generator=torch.Generator().manual_seed(42))
        loader=DataLoader(TensorDataset(t,s,y),batch_size=32,sampler=sampler)
        root='iterations_v2' if name=='whiten' else 'iterations_tf'
        selected=[json.loads(Path(f'artifacts/{root}/{name}_{r}_seed42/metrics.json').read_text())['best_epoch'] for r in ['a1','c1','g1']]
        epochs=int(np.median(selected))
        config=dict(name=name,seed=42,fixed_epochs=epochs,fold_best_epochs=selected,lr=.0007,weight_decay=.01,
                    label_smoothing=.1,epoch_selection='median of development fold best epochs',
                    lr_schedule='constant during fixed-length full refit',unknown_grouping='one trace proxy per unknown identifier, not asserted physical room')
        configs.append(config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
        criterion=nn.CrossEntropyLoss(label_smoothing=.1);history=[]
        for epoch in range(1,epochs+1):
            model.train();total=correct=0;loss_sum=0.
            for a,b,c in loader:
                a,b,c=a.cuda(),b.cuda(),c.cuda();optimizer.zero_grad(set_to_none=True)
                logits=model.forward_features(a,b);loss=criterion(logits,c);assert torch.isfinite(loss)
                loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),3.);assert torch.isfinite(norm)
                optimizer.step();total+=len(c);correct+=(logits.argmax(1)==c).sum().item();loss_sum+=loss.item()*len(c)
            history.append(dict(epoch=epoch,training_objective=loss_sum/total,sampled_training_accuracy=correct/total))
            print('Full refit',name,epoch,'/',epochs,'sampled train accuracy',correct/total,flush=True)
        histories[name]=history
        fusion.models[MEMBERS.index(name)].load_state_dict(model.cpu().state_dict())
        del model,t,s,y,bank,extra_t,extra_s
    fusion.eval()
    config=dict(members=configs,records=48,windows=960,external_data=False,unlabeled_data=False,
                independent_validation_accuracy=None,development_scores_belong_to_separate_holdout_models=True,
                source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['roomsense_model.py','improved_models.py','tf_model.py','fusion_model.py','train_submission.py']})
    (out/'config.json').write_text(json.dumps(config,indent=2));(out/'history.json').write_text(json.dumps(histories,indent=2))
    torch.save(dict(model_state=fusion.state_dict(),config=config),out/'final.pt')
    first=None
    wrapper=SubmissionModel(fusion)
    for name in ['train','heldout','gap','zero']:
        with np.load(f'artifacts/model_design/case_{name}.npz') as data:x,dt=data['frames'].copy(),data['intervals'].copy()
        with torch.no_grad():reference=wrapper(torch.from_numpy(x),torch.from_numpy(dt)).numpy()
        np.savez(out/f'case_{name}.npz',frames=x,intervals=dt,logits=reference)
        if first is None:first=(torch.from_numpy(x),torch.from_numpy(dt))
    export_submission(fusion,first,out/'model.onnx')
    import onnx
    onnx.checker.check_model(str(out/'model.onnx'));assert (out/'model.onnx').stat().st_size<100_000_000
    subprocess.run([sys.executable,'-X','utf8','validate_model.py','--cpu-only','--output',str(out),'--model-name','model.onnx'],check=True)
    print('Full-label submission candidate ready; no independent local accuracy claimed:',out/'model.onnx',flush=True)


if __name__=='__main__':main()
