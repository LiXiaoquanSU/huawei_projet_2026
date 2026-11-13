"""Recover distinct direct/ordinal/route opinions from the same frozen network."""
from arbiter_common import *

def internal_opinions(tasks):
    return torch.cat([tasks['direct'][:,None],tasks['ordinal'][:,None],tasks['experts'].softmax(-1)],1)

def pooled_aux(extra,b,T):
    q=(extra.mean(2).clamp_min(1e-12).log()/T+b+extra.new_tensor([0,.35,0,0,0])).exp().mean(1)
    return q/q.sum(-1,keepdim=True).clamp_min(1e-12)

def predict(model,tt,ss,indices):
    model.eval();views=ReceiverViews().cuda();raw=[];extra=[]
    with torch.inference_mode():
        for start in range(0,len(indices),4):
            ix=indices[start:start+4];t,s=views(tt[ix].flatten(0,1),ss[ix].flatten(0,1));tasks=model.forward_tasks(t,s)
            raw.append(tasks['logits'].reshape(6,len(ix),2,5).permute(1,0,2,3).cpu())
            extra.append(internal_opinions(tasks).reshape(6,len(ix),2,6,5).permute(1,0,2,3,4).cpu())
    return torch.cat(raw),torch.cat(extra)

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=WORK/'aux_heads';out.mkdir(exist_ok=True)
    tt=torch.from_numpy(np.array(np.load(PREV/'t.npy',mmap_mode='r'),copy=True)).cuda();ss=torch.from_numpy(np.array(np.load(PREV/'s.npy',mmap_mode='r'),copy=True)).cuda()
    for room in (*ROOMS,'actual'):
        path=out/f'oof_{room}.pt'
        if not path.exists():
            parts=[]
            for fold in range(3):
                folder=WORK/'crossfit'/room/f'fold{fold}';p=torch.load(folder/'predictions.pt',weights_only=True)
                cache=out/f'aux_{room}_fold{fold}.pt'
                if cache.exists():v=torch.load(cache,weights_only=True)
                else:
                    model,_=old.build(folder/'final.pt');model=model.cuda().eval();raw,aux=predict(model,tt,ss,p['indices'])
                    error=float((raw-p['raw']).abs().max());assert error<1e-4
                    probs=torch.cat([p['probabilities'],pooled_aux(aux,p['bias'],p['temperature'])],1)
                    v=dict(probabilities=probs,raw_error=error);torch.save(v,cache)
                    del model;gc.collect();torch.cuda.empty_cache()
                parts.append(v['probabilities']);print('AUX_OOF',room,fold,flush=True)
            d=torch.load(WORK/f'oof_{room}.pt',weights_only=True);d['probabilities']=torch.cat(parts);torch.save(d,path)
        path=out/f'target_{room}.pt'
        if not path.exists():
            d=torch.load(EVID/f'{room}.pt',weights_only=True);ix=old.split(room,d['rows'])[2]
            model,_=old.build(old.checkpoint(room),full=room=='actual');model=model.cuda().eval();raw,extra=predict(model,tt,ss,ix);b,T=old.calibration(room)
            error=float((raw-d['raw'][ix]).abs().max());assert error<1e-4
            p=torch.cat([opinions(raw,b,T),pooled_aux(extra,b,T)],1)
            # Targets are held opaque here; labels used by evaluator only after training.
            torch.save(dict(probabilities=p,indices=ix,raw_error=error),path)
            del model;gc.collect();torch.cuda.empty_cache();print('AUX_TARGET',room,flush=True)
    save(out/'complete.json',dict(opinions=26,extra_heads=['direct','ordinal','static','dynamic','phase','time'],
      calibration='Same original perteacher bias/temperature, not independently tuned head biases.',new_teacher_training=False))

if __name__=='__main__':main()
