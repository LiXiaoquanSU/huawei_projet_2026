"""Attach selected invariant route residual to exact scored selective054 backbone."""
import onnx
from onnx import helper,numpy_helper,TensorProto
from ra_common import *
from export_batch import FeatureHead,NAMES

class RouteHead(FeatureHead):
    def __init__(self,selection):
        super().__init__('transfer_low_guard');self.amount=selection['amount'];heads=[]
        for seed in selection['seeds']:
            dest=WORK/'fits'/f"{selection['architecture']}_{selection['objective']}_{seed}_actual"
            step=read(dest/'complete.json')['selected_step'];heads.append(model_from(torch.load(dest/f'step{step}.pt',weights_only=True)))
        self.residuals=nn.ModuleList(heads)
    def forward(self,t0,s0,t1,s1):
        t,s=self.views(torch.cat([t0,t1]),torch.cat([s0,s1]));raw=self.model.forward_features(t,s).reshape(1,6,2,5)
        e=torch.stack([self.captured[k] for k in ('static','dynamic','phase','temporal')],1)
        student,_=self.student(self.captured['p'],self.captured['t'],e,s)
        base=prev.transfer(old.policy(raw,self.bias),old.policy(student.reshape(1,6,2,5),self.bias),'low_guard')
        stats=moments(e[:,1:].reshape(1,12,3,256));aux=auxiliary(raw,self.bias,1.)
        delta=torch.stack([h(stats,base,aux) for h in self.residuals]).mean(0)
        self.last_base=base[0];return deploy(base,delta,self.amount)[0]

class Pair(nn.Module):
    def __init__(self,h):super().__init__();self.h=h
    def forward(self,*args):
        out=self.h(*args);return out,self.h.last_base

def main():
    selection=read(WORK/'selection.json');assert selection['eligible'];folder=WORK/'export';folder.mkdir(exist_ok=True);torch.set_num_threads(4)
    head=RouteHead(selection).eval();examples=(torch.zeros(1,963,150),torch.zeros(1,17),torch.zeros(1,963,150),torch.zeros(1,17))
    torch.onnx.export(Pair(head),examples,str(folder/'head.onnx'),input_names=NAMES,output_names=['ra_logits','ra_control'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(folder/'head.onnx');casts=[];compressed=[]
    for tensor in small.graph.initializer:
        a=numpy_helper.to_array(tensor)
        if a.dtype==np.float32 and a.size>=2048 and np.isfinite(a).all() and np.array_equal(a,a.astype(np.float16).astype(np.float32)):
            name=tensor.name;new=name+'__exact_fp16';tensor.CopyFrom(numpy_helper.from_array(a.astype(np.float16),new));casts.append(helper.make_node('Cast',[new],[name],to=TensorProto.FLOAT));compressed.append(name)
    nodes=list(small.graph.node);del small.graph.node[:];small.graph.node.extend(casts+nodes)
    mapping={t.name:'__ra_'+t.name for t in small.graph.initializer}
    for n in small.graph.node:
        for value in n.output:mapping[value]='logits' if value=='ra_logits' else '__ra_control' if value=='ra_control' else '__ra_'+value
    for t in small.graph.initializer:t.name=mapping[t.name]
    for node_index,n in enumerate(small.graph.node):
        n.name='__ra_'+(n.name or f'unnamed_node_{node_index}')
        for i,value in enumerate(n.input):n.input[i]=mapping.get(value,value)
        for i,value in enumerate(n.output):n.output[i]=mapping.get(value,value)
    m=onnx.load(PREV/'preprocessing.onnx');m.graph.node.extend(small.graph.node);m.graph.initializer.extend(small.graph.initializer);del m.graph.output[:];m.graph.output.append(helper.make_tensor_value_info('logits',TensorProto.FLOAT,[5]))
    helper.set_model_props(m,dict(base_sha256=SOURCE_SHA,base_platform_accuracy='0.54 user reported',platform_accuracy='not evaluated',method='frozen054 invariant-view route residual',selection=json.dumps(selection),protection='current054 final argmax>=2 returns original full vector',preprocessing='Exact original054 extracted nodes',storage='Exactly representable tensors only compressed toFP16; float32 compute'))
    onnx.checker.check_model(m);onnx.save(m,folder/'model.onnx')
    save(folder/'structure_proof.json',dict(candidate_sha256=sha(folder/'model.onnx'),base_sha256=SOURCE_SHA,preprocessor_sha256=sha(PREV/'preprocessing.onnx'),bytes=(folder/'model.onnx').stat().st_size,selection=selection,compressed=compressed,new_parameters=sum(p.numel() for h in head.residuals for p in h.parameters()),existing_backbone_frozen=True))
    print('RA_EXPORTED',sha(folder/'model.onnx'),(folder/'model.onnx').stat().st_size,flush=True)
if __name__=='__main__':main()
