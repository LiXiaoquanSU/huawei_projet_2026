"""Independent NumPy reference from original048's existing evidence tensors."""
import gc
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,TensorProto
import torch
from export_learned import ROOT,WORK,EXPORT,BASE,HEAD,PROB,Head,audit


def main():
    torch.set_num_threads(4)
    cp=torch.load(WORK/'full_data/pair0/final.pt',map_location='cpu',weights_only=True)
    weights,bias=cp['weights'].numpy(),cp['bias'].numpy()
    names=audit.read(EXPORT/'structure_proof.json')['reused_tensors']
    original=onnx.load(audit.SOURCE)
    for i,name in enumerate(names):
        shape=[1,4,5] if i%3==2 else [1,5]
        original.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shape))
    engine=audit.engine(original.SerializeToString());del original;gc.collect()
    refs=audit.read(EXPORT/'references.json');errors=[]
    for reference in refs:
        if reference['input']=='zero':inputs=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
        else:
            with np.load(reference['input']) as d:inputs={k:d[k].copy() for k in ('frames','intervals')}
        outputs=engine.run(None,inputs);base=outputs[0];a=[];b=[]
        for k in range(4):
            for values in (outputs[1+k*3],outputs[2+k*3],outputs[3+k*3][:,0]):
                p=np.maximum(values,np.float32(1e-6))
                a.append(np.clip(np.log(p[:,3]+p[:,4])-np.log(p[:,2]),-6,6))
                b.append(np.clip(np.log(p[:,4])-np.log(p[:,3]),-6,6))
        x=np.stack([np.stack(a,1),np.stack(b,1)],1)
        z=(x*weights[None]).sum(-1)+bias
        q,r=(1/(1+np.exp(-z)))[0]
        probability=np.array([0,0,1-q,q*(1-r),q*r],np.float32)
        logits=np.concatenate([np.full(2,-100,np.float32),np.log(np.maximum(probability[2:],np.float32(1e-8)))])
        winner=int(probability.argmax())
        trigger=int(base.argmax())>=2 and winner<int(base.argmax()) and float(probability.max())>=float(np.float32(.6))
        expected=logits if trigger else base
        error=float(np.abs(expected-np.array(reference['logits'],np.float32)).max());assert error<2e-5,error
        errors.append(dict(input=reference['input'],trigger=trigger,independent_logit_error=error))
        reference['logits']=expected.tolist();reference['reference_method']='original048 graph evidence + independent NumPy26-parameter conditional head'
    del engine;gc.collect()
    audit.save(EXPORT/'references.json',refs)
    audit.save(EXPORT/'independent_reference_check.json',dict(cases=errors,max_logit_error=max(r['independent_logit_error'] for r in errors),
        original_sha256=audit.sha(audit.SOURCE),head_checkpoint_sha256=audit.sha(WORK/'full_data/pair0/final.pt')))
    print('INDEPENDENT_REFERENCES_PASS',len(errors),max(r['independent_logit_error'] for r in errors),flush=True)


if __name__=='__main__':main()
