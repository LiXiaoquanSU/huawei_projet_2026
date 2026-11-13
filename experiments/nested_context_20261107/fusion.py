"""Source-OOF calibrated mixture; fixed regularization, no outer-label fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

W=Path('D:/huawei2026-experiments/nested_context_20261107')
OUT=Path(__file__).resolve().parent
torch.set_num_threads(4)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def load(p): return torch.load(p,map_location='cpu',weights_only=False)
def calibrate(z,y):
    # Fixed L2 penalty and optimizer budget; all fitting uses source OOF only.
    b=torch.zeros(5,requires_grad=True); lt=torch.zeros((),requires_grad=True)
    opt=torch.optim.LBFGS([b,lt],lr=.3,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); logits=z/lt.clamp(-1.4,1.4).exp()+b-b.mean()
        loss=torch.nn.functional.cross_entropy(logits,y)+.05*b.square().mean()+.02*lt.square()
        loss.backward(); return loss
    opt.step(closure)
    return b.detach()-b.detach().mean(),lt.detach().clamp(-1.4,1.4).exp()
def features(a,b):
    def f(p):
        top=p.topk(2,dim=1).values
        return torch.cat([p,(-p*p.clamp_min(1e-8).log()).sum(1,keepdim=True),top[:,:1]-top[:,1:]],1)
    return torch.cat([f(a),f(b),(a-b).abs()],1).numpy()
def main():
    results=[]; predictions=[]
    for criterion in ['accuracy','fixed5']:
        for fold in range(3):
            source=[]; target=[]; calibration=[]
            for variant in ['baseline','context_events']:
                sel=read(W/f'record{fold}_{variant}_selection.json')
                epoch=sel['accuracy_epoch']-1 if criterion=='accuracy' else 4
                d=load(W/f'record{fold}_{variant}_oof.pt'); order=np.argsort(d['indices'])
                ix=np.array(d['indices'])[order]; y=d['labels'][order]; z=d['logits'][epoch,order]
                q=load(W/f'record{fold}_{variant}_outer/predictions.pt')
                if source: assert np.array_equal(ix,source_ix) and torch.equal(y,sy) and q['indices']==test_ix
                source_ix=ix;sy=y;test_ix=q['indices'];ty=q['labels']
                bias,temp=calibrate(z,y); source.append((z/temp+bias).softmax(1));target.append((q['logits'][epoch]/temp+bias).softmax(1))
                calibration.append(dict(variant=variant,epoch=epoch+1,bias=bias.tolist(),temperature=float(temp)))
            a,b=source; ta,tb=target
            ca=a.argmax(1)==sy;cb=b.argmax(1)==sy; mask=ca!=cb
            x=features(a,b);xt=features(ta,tb);scaler=StandardScaler().fit(x[mask])
            gate=LogisticRegression(C=.1,max_iter=1000,random_state=42).fit(scaler.transform(x[mask]),cb[mask].numpy())
            w=torch.from_numpy(gate.predict_proba(scaler.transform(xt))[:,1]).float()[:,None]
            # Conservative fixed bounds avoid an overconfident selector replacing either expert.
            w=w.clamp(.25,.75)
            variants={'baseline_cal':ta,'context_cal':tb,'equal_cal':(ta+tb)/2,'gate_cal':ta*(1-w)+tb*w}
            base=ta.argmax(1)==ty; other=tb.argmax(1)==ty
            comp=dict(both_correct=int((base&other).sum()),baseline_only=int((base&~other).sum()),context_only=int((~base&other).sum()),both_wrong=int((~base&~other).sum()))
            for name,p in variants.items():
                pred=p.argmax(1);cm=torch.bincount(ty*5+pred,minlength=25).reshape(5,5)
                r=dict(criterion=criterion,fold=fold,method=name,correct=int(cm.trace()),total=len(ty),confusion=cm.tolist(),complementarity=comp,calibration=calibration)
                results.append(r);print(criterion,fold,name,r['correct'],flush=True)
                predictions.append(dict(criterion=criterion,fold=fold,method=name,indices=test_ix,labels=ty,probabilities=p))
    totals={}
    for r in results:
        key=r['criterion']+'/'+r['method'];totals[key]=totals.get(key,0)+r['correct']
    (OUT/'fusion_results.json').write_text(json.dumps(dict(total=960,totals=totals,results=results),indent=2)+'\n',encoding='utf-8')
    torch.save(predictions,OUT/'fusion_predictions.pt');print(totals)
if __name__=='__main__':main()
