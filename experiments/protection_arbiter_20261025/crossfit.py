"""Generate nested record-OOF reliability examples, excluding each outer room."""
from arbiter_common import *
from model import StructuredModel,signal_targets
from utils import task_loss

def partition(rows,indices):
    records=sorted({rows[i]['identifier'] for i in indices})
    folds={}
    for label in range(5):
        ids=[r for r in records if next(int(v['headcount']) for v in rows if v['identifier']==r)==label]
        ids.sort(key=lambda x:hashlib.sha256(('arbiter-oof-v1:'+x).encode()).hexdigest())
        assert len(ids)>=2,(label,len(ids))
        for j,r in enumerate(ids):folds[r]=j%3
    return [[i for i in indices if folds[rows[i]['identifier']]==k] for k in range(3)]

def actual_split(rows):
    chosen=[]
    for label in range(5):
        ids={r['identifier'] for r in rows if int(r['headcount'])==label}
        chosen.append(min(ids,key=lambda x:hashlib.sha256(('calibration17:'+x).encode()).hexdigest()))
    return ([i for i,r in enumerate(rows) if r['identifier'] not in chosen],
            [i for i,r in enumerate(rows) if r['identifier'] in chosen],[])

def calibrate(raw,y):
    # Bias and temperature fitted on separate source calibration records only.
    bias=torch.zeros(5,device=raw.device,requires_grad=True);temp=torch.zeros((),device=raw.device,requires_grad=True)
    optimizer=torch.optim.Adam([bias,temp],lr=.03)
    w=1/torch.bincount(y,minlength=5).clamp_min(1)[y].float();w=w/w.sum()
    for _ in range(150):
        b=2*bias.tanh();b=b-b.mean();T=temp.exp().clamp(.5,3)
        loss=(F.cross_entropy(raw[:,0].exp().mean(1).clamp_min(1e-12).log()/T+b,y,reduction='none')*w).sum()
        loss=loss+.05*b.square().mean()+.05*temp.square()
        optimizer.zero_grad();loss.backward();optimizer.step()
    b=2*bias.detach().tanh();b=b-b.mean()
    return b,float(temp.detach().exp().clamp(.5,3))

def sample_weights(rows,y,fit):
    groups={};key=lambda i:(int(y[i]),rows[i]['identifier'] if rows[i]['room']=='unknown' else rows[i]['room'])
    for i in fit:groups[key(i)]=groups.get(key(i),0)+1
    counts={label:sum(a==label for a,b in groups) for label in range(5)}
    return torch.tensor([1/(groups[key(i)]*counts[int(y[i])]) for i in fit])

def predict(model,tt,ss,indices):
    model.eval();views=ReceiverViews().cuda();out=[]
    with torch.inference_mode():
        for start in range(0,len(indices),4):
            ix=indices[start:start+4];t,s=views(tt[ix].flatten(0,1),ss[ix].flatten(0,1))
            z=model.forward_features(t,s).reshape(6,len(ix),2,5).permute(1,0,2,3)
            out.append(z.cpu())
    return torch.cat(out)

def fit(room,fold,fit,cal,test,rows,tt,ss,y):
    folder=WORK/'crossfit'/room/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'predictions.pt').exists():return
    trainids={rows[i]['identifier'] for i in fit};calids={rows[i]['identifier'] for i in cal};testids={rows[i]['identifier'] for i in test}
    assert not trainids&calids and not trainids&testids and not calids&testids
    if room!='actual':assert all(rows[i]['room']!=room for i in fit+cal+test)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=StructuredModel('spectral','physical').cuda()
    sampler=torch.Generator().manual_seed(42);weights=sample_weights(rows,y,fit)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01)
    target=signal_targets(tt[:,0]);history=[];fit_tensor=torch.tensor(fit,device='cuda')
    for epoch in range(5):
        draws=torch.multinomial(weights,len(fit),replacement=True,generator=sampler);ix=fit_tensor[draws.cuda()]
        model.train();losses=[]
        for start in range(0,len(ix),24):
            jj=ix[start:start+24];optimizer.zero_grad(set_to_none=True)
            tasks=model.forward_tasks(tt[jj,0],ss[jj,0]);loss,_=task_loss(tasks,y[jj],target[jj],'physical')
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses))))
    torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=42,epochs=5,
                    fit_records=sorted(trainids),calibration_records=sorted(calids),oof_records=sorted(testids),history=history),folder/'final.pt')
    # Same half-storage keys as the submitted original, not arbitrary weight rounding.
    keys=old.build.__globals__['precision_spec']()[0]
    with torch.no_grad():
        state=model.state_dict()
        for key in keys:state[key].copy_(state[key].half().float())
    rawcal=predict(model,tt,ss,cal);b,T=calibrate(rawcal.cuda(),y[cal]);raw=predict(model,tt,ss,test)
    torch.save(dict(raw=raw,labels=y[test].cpu(),indices=test,rows=[rows[i] for i in test],bias=b.cpu(),temperature=T,
                    fit_records=sorted(trainids),calibration_records=sorted(calids),oof_records=sorted(testids),
                    probabilities=opinions(raw,b.cpu(),T)),folder/'predictions.pt')
    save(folder/'manifest.json',dict(room=room,fold=fold,fit_records=sorted(trainids),calibration_records=sorted(calids),oof_records=sorted(testids),
       seed=42,epochs=5,batch_size=24,lr=.0004,weight_decay=.01,training_preprocessing='Actual054 FLOAT32 ONNX feature cache; 1 identity view',
       calibration='Source-only separate records, fixed150steps bias/temperature',checkpoint_sha256=sha(folder/'final.pt')))
    print('CROSSFIT_DONE',room,fold,len(fit),len(cal),len(test),history[-1],flush=True)
    del model,optimizer,target;gc.collect();torch.cuda.empty_cache()

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rooms=['actual'] if '--actual' in sys.argv else ROOMS
    rows=read(PREV/'rows.json');y=torch.tensor([int(r['headcount']) for r in rows],device='cuda')
    tt=torch.from_numpy(np.array(np.load(PREV/'t.npy',mmap_mode='r'),copy=True)).cuda()
    ss=torch.from_numpy(np.array(np.load(PREV/'s.npy',mmap_mode='r'),copy=True)).cuda()
    WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(method='Nested record-OOF relative utility arbiter. Original054 default, no targetlabels in fitting or checkpoint selection.',
      parent=old.SOURCE_SHA,nested_folds=3,teacher_epochs=5,teacher_seed=42,teacher_batch_size=24,teacher_lr=.0004,
      outer_rooms=list(ROOMS),gate_seeds=[42,2026],features='Classagnostic entropy/margin/support/disagreement, optional candidate-label onehot.',
      source_selection='Record-disjoint fit/calibration, gate OOF fit/validation, fixed candidates selected before outer target evaluation.',
      outer_limit='Repeated development rooms; not a new blind test. No platform-score claim.'))
    for room in rooms:
        source,cal,_=actual_split(rows) if room=='actual' else old.split(room,rows)
        folds=partition(rows,source)
        for k,test in enumerate(folds):fit(room,k,[i for i in source if i not in set(test)],cal,test,rows,tt,ss,y)
        parts=[torch.load(WORK/'crossfit'/room/f'fold{k}/predictions.pt',weights_only=True) for k in range(3)]
        ix=sum([p['indices'] for p in parts],[]);assert sorted(ix)==sorted(source) and len(ix)==len(set(ix))
        torch.save(dict(probabilities=torch.cat([p['probabilities'] for p in parts]),labels=torch.cat([p['labels'] for p in parts]),
                        rows=sum([p['rows'] for p in parts],[]),indices=ix,folds=sum([[k]*len(p['indices']) for k,p in enumerate(parts)],[])),WORK/f'oof_{room}.pt')
    save(WORK/('actual_crossfit_complete.json' if '--actual' in sys.argv else 'proxy_crossfit_complete.json'),dict(rooms=list(rooms)))

if __name__=='__main__':main()
