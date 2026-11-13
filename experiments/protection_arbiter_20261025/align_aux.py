"""Calibrate EACH auxiliary head to its teacher marginal on source cal records."""
from auxiliary import *
from crossfit import actual_split

def fit_delta(prob,target):
    logp=prob.clamp_min(1e-12).log();delta=torch.zeros(6,5)
    for _ in range(400):
        mean=(logp+delta).softmax(-1).mean(0).clamp_min(1e-10)
        delta+=.5*(target.clamp_min(1e-10).log()[None]-mean.log())
        delta-=delta.mean(-1,keepdim=True)
    error=float(((logp+delta).softmax(-1).mean(0)-target).abs().max());assert error<.002,error
    return delta,error

def apply_delta(p,delta):
    return torch.cat([p[:,:20],(p[:,20:].clamp_min(1e-12).log()+delta).softmax(-1)],1)

def calibration(path,checkpoint,cal,b,T,tt,ss):
    if path.exists():return torch.load(path,weights_only=True)['delta']
    model,_=old.build(checkpoint);model=model.cuda().eval();raw,extra=predict(model,tt,ss,cal)
    base=old.policy(raw,b,T).softmax(-1);p=pooled_aux(extra,b,T)
    label_cal='--label-cal' in sys.argv
    target=torch.bincount(torch.tensor([int(ROWS[i]['headcount']) for i in cal]),minlength=5).float()/len(cal) if label_cal else base.mean(0)
    delta,error=fit_delta(p,target)
    torch.save(dict(delta=delta,calibration_indices=cal,calibration_records=sorted({ROWS[i]['identifier'] for i in cal}),
                    target_teacher_marginal=base.mean(0),target_marginal=target,calibration_probabilities=p,matching_error=error,labels_used=label_cal),path)
    del model;gc.collect();torch.cuda.empty_cache();return delta

def main():
    global ROWS
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=WORK/('aux_labelcal' if '--label-cal' in sys.argv else 'aux_aligned');out.mkdir(exist_ok=True);src=WORK/'aux_heads';ROWS=read(PREV/'rows.json')
    tt=torch.from_numpy(np.array(np.load(PREV/'t.npy',mmap_mode='r'),copy=True)).cuda();ss=torch.from_numpy(np.array(np.load(PREV/'s.npy',mmap_mode='r'),copy=True)).cuda()
    for room in (*ROOMS,'actual'):
        if not (out/f'oof_{room}.pt').exists():
            parts=[]
            for fold in range(3):
                folder=WORK/'crossfit'/room/f'fold{fold}';p=torch.load(folder/'predictions.pt',weights_only=True)
                cal=[i for i,r in enumerate(ROWS) if r['identifier'] in set(p['calibration_records'])]
                delta=calibration(out/f'cal_{room}_fold{fold}.pt',folder/'final.pt',cal,p['bias'],p['temperature'],tt,ss)
                v=torch.load(src/f'aux_{room}_fold{fold}.pt',weights_only=True);parts.append(apply_delta(v['probabilities'],delta))
                print('AUX_ALIGN',room,fold,flush=True)
            d=torch.load(src/f'oof_{room}.pt',weights_only=True);d['probabilities']=torch.cat(parts);torch.save(d,out/f'oof_{room}.pt')
        if not (out/f'target_{room}.pt').exists():
            cal=actual_split(ROWS)[1] if room=='actual' else old.split(room,ROWS)[1];b,T=old.calibration(room)
            delta=calibration(out/f'cal_target_{room}.pt',old.checkpoint(room),cal,b,T,tt,ss)
            d=torch.load(src/f'target_{room}.pt',weights_only=True);d['probabilities']=apply_delta(d['probabilities'],delta);torch.save(d,out/f'target_{room}.pt')
            print('AUX_ALIGN_TARGET',room,flush=True)
    save(out/'complete.json',dict(method='Per-head additive calibration on separate SOURCE cal records only. Actual calibration class proportions iflabelcal; otherwise teacher marginal. No uniform-per-room constraint.',
          direct_labels_used='--label-cal' in sys.argv,original054bias_changed=False,auxiliary_head_biases=True,steps=400))

if __name__=='__main__':main()
