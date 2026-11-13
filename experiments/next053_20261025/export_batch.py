"""Shared exact054 preprocessing plus one batched CNN for12/24 views."""
import gc,time,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from n53_common import *
from common import ReceiverViews
sys.path.insert(0,str(ROOT/'experiments/u_ssl_20261025'))
from ssl_common import Student
from quantized import updated_state

NAMES=[f'__rxg_012___tta_v{t}_{suffix}' for t in (0,1) for suffix in ('cat_8','cat_6')]
class FeatureHead(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method;self.model,b=old.build(old.checkpoint('actual'),full=True);self.model.eval();self.register_buffer('bias',b);self.views=ReceiverViews();self.captured={}
        if method.startswith('transfer_'):
            self.student=Student(self.model).eval();state,_=updated_state('actual');self.student.load_state_dict({k.replace('phase.project.','phase_project.').replace('temporal.blocks.2.','time_block.').replace('temporal.project.','time_project.'):v for k,v in state.items()},strict=False)
            self.model.phase.project.register_forward_pre_hook(self.pre('p'));self.model.temporal.blocks[-1].register_forward_pre_hook(self.pre('t'))
            for name in ('static','dynamic','phase','temporal'):getattr(self.model,name).register_forward_hook(self.post(name))
    def pre(self,name):
        def hook(m,args):self.captured[name]=args[0]
        return hook
    def post(self,name):
        def hook(m,args,out):self.captured[name]=out
        return hook
    def forward(self,t0,s0,t1,s1):
        t,s=self.views(torch.cat([t0,t1]),torch.cat([s0,s1]))
        if self.method.startswith('mirror'):
            both=self.model.forward_features(torch.cat([t,mirror_features(t)]),torch.cat([s,s]));raw=both[:12].reshape(1,6,2,5);mirrored=both[12:].reshape(1,6,2,5)
            self.last_base=old.policy(raw,self.bias)[0]
            return spectral(raw,mirrored,self.bias,1.,.25 if self.method=='mirror25' else .5)[0]
        raw=self.model.forward_features(t,s).reshape(1,6,2,5)
        self.last_base=old.policy(raw,self.bias)[0]
        if self.method.startswith('transfer_'):
            e=torch.stack([self.captured[k] for k in ('static','dynamic','phase','temporal')],1);z,_=self.student(self.captured['p'],self.captured['t'],e,s);adapted=old.policy(z.reshape(1,6,2,5),self.bias)
            return transfer(old.policy(raw,self.bias),adapted,self.method.removeprefix('transfer_'))[0]
        return pool(raw,self.bias,1.,self.method)[0]

class ExportPair(nn.Module):
    def __init__(self,core):super().__init__();self.core=core
    def forward(self,*args):
        out=self.core(*args);return out,self.core.last_base

def assemble(method,folder):
    folder.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);head=FeatureHead(method).eval();examples=(torch.zeros(1,963,150),torch.zeros(1,17),torch.zeros(1,963,150),torch.zeros(1,17))
    torch.onnx.export(ExportPair(head),examples,str(folder/'head.onnx'),input_names=NAMES,output_names=['n53_logits','n53_control'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(folder/'head.onnx');compressed=[];casts=[]
    for tensor in small.graph.initializer:
        a=numpy_helper.to_array(tensor)
        if a.dtype==np.float32 and a.size>=2048 and np.isfinite(a).all() and np.array_equal(a,a.astype(np.float16).astype(np.float32)):
            oldname=tensor.name;newname=oldname+'__exact_fp16';tensor.CopyFrom(numpy_helper.from_array(a.astype(np.float16),name=newname));casts.append(helper.make_node('Cast',[newname],[oldname],to=TensorProto.FLOAT,name='n53_exact_storage_'+oldname));compressed.append(oldname)
    original_nodes=list(small.graph.node);del small.graph.node[:];small.graph.node.extend(casts+original_nodes)
    mapping={t.name:'__n53_'+t.name for t in small.graph.initializer}
    for n in small.graph.node:
        for value in n.output:mapping[value]='logits' if value=='n53_logits' else '__n53_control' if value=='n53_control' else '__n53_'+value
    for t in small.graph.initializer:t.name=mapping[t.name]
    for n in small.graph.node:
        assert not any(a.type==onnx.AttributeProto.GRAPH for a in n.attribute)
        n.name='__n53_'+n.name
        for i,value in enumerate(n.input):n.input[i]=mapping.get(value,value)
        for i,value in enumerate(n.output):n.output[i]=mapping.get(value,value)
    model=onnx.load(WORK/'preprocessing.onnx');model.graph.node.extend(small.graph.node);model.graph.initializer.extend(small.graph.initializer);del model.graph.output[:];model.graph.output.append(helper.make_tensor_value_info('logits',TensorProto.FLOAT,[5]))
    helper.set_model_props(model,dict(base_sha256=SOURCE_SHA,base_platform_accuracy='0.54 userreported',platform_accuracy='not evaluated',method=method,
      preprocessing='Exact extracted054identity/time-reverseONNX nodes; receiver feature permutations proven exact on8rawwindows includingcase599.',
      inference='One batched12view network,24forfrequency-mirror. Originalprecision-matched044weights unchanged unless explicittransferexperiment.',
      weight_storage='Only exactly-fp16-representable arrays compressed; castbackFLOAT is lossless relative to the Torch reference.',caveat='Local development controls are not hidden platform accuracy.'))
    onnx.checker.check_model(model);onnx.save(model,folder/'model.onnx')
    proof=dict(method=method,candidate_sha256=sha(folder/'model.onnx'),source_sha256=SOURCE_SHA,preprocessor_sha256=sha(WORK/'preprocessing.onnx'),inference_views=24 if method.startswith('mirror') else 12,
      exact_fp16_storage_tensors=compressed,bytes=(folder/'model.onnx').stat().st_size,parameters_updated=method.startswith('transfer_'),no_new_bias=True)
    save(folder/'structure_proof.json',proof);print('N53_BATCH_EXPORTED',method,proof['bytes'],len(compressed),flush=True);return head

def prototype():
    folder=WORK/'prototype';head=assemble('baseline',folder);session=make_session(folder/'model.onnx');prep=make_session(WORK/'preprocessing.onnx');cases=[]
    meta=read(ROOT/'models/teammate054_20261025/gprime_full_group_fallback_web054.json')
    with torch.inference_mode():
        for original in meta['audit']['formula_probes'][:3]:
            if original['input']=='zero':feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
            else:
                with np.load(original['input']) as d:feed={k:d[k].copy() for k in ('frames','intervals')}
            features=prep.run(None,feed);reference=head(*[torch.from_numpy(a) for a in features]).numpy();out=session.run(None,feed)[0];oldref=np.array(original['logits']);e=float(np.abs(out-oldref).max());assert e<.002,e
            times=[]
            for i in range(5):start=time.perf_counter();session.run(None,feed);times.append(time.perf_counter()-start)
            cases.append(dict(input=original['input'],old054_maxlogit_error=e,torch_maxlogit_error=float(np.abs(out-reference).max()),rough_cpu_mean_s=float(np.mean(times))))
    save(folder/'prototype.json',cases);print('N53_BATCH_PROTOTYPE',cases,flush=True)
if __name__=='__main__':prototype()
