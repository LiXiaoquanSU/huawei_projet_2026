"""Publish only the prior-free, cross-room qualified candidate."""
import shutil
import onnx
from onnx import helper,numpy_helper,TensorProto
from bayes import *

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.model,b=old.build(old.checkpoint('actual'),full=True);self.register_buffer('bias',b);self.views=ReceiverViews()
        cp=torch.load(WORK/'actual.pt',weights_only=True);self.config=cp['config'];self.gates=nn.ModuleList()
        for state in cp['states']:
            m=Gate();m.load_state_dict(state);self.gates.append(m)
    def forward(self,t0,s0,t1,s1):
        t,s=self.views(torch.cat([t0,t1]),torch.cat([s0,s1]));raw=self.model.forward_features(t,s).reshape(1,6,2,5)
        p,f,e=data(raw,self.bias,1.)
        score=torch.stack([gate(f) for gate in self.gates]).mean(0)
        return choose(p,score,e,self.config['threshold'],self.config['amount'])[0][0]

def assemble():
    torch.set_num_threads(4);r=read(WORK/'results.json');assert r['passed']
    a=read(WORK/'actual_metrics.json');assert a['correct']>=647
    d=torch.load(WORK/'eval_actual.pt',weights_only=True);torch.save(d,WORK/'actual_predictions.pt')
    head=Head().eval();folder=WORK/'export';folder.mkdir(exist_ok=True)
    names=[f'__rxg_012___tta_v{t}_{suffix}' for t in (0,1) for suffix in ('cat_8','cat_6')]
    sample=(torch.zeros(1,963,150),torch.zeros(1,17),torch.zeros(1,963,150),torch.zeros(1,17))
    torch.onnx.export(head,sample,str(folder/'head.onnx'),input_names=names,output_names=['priorfree_logits'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(folder/'head.onnx');casts=[]
    for tensor in small.graph.initializer:
        value=numpy_helper.to_array(tensor)
        if value.dtype==np.float32 and value.size>=2048 and np.array_equal(value,value.astype(np.float16).astype(np.float32)):
            name=tensor.name;half=name+'__fp16';tensor.CopyFrom(numpy_helper.from_array(value.astype(np.float16),name=half));casts.append(helper.make_node('Cast',[half],[name],to=TensorProto.FLOAT))
    nodes=list(small.graph.node);del small.graph.node[:];small.graph.node.extend(casts+nodes)
    mapping={x.name:'__priorfree_'+x.name for x in small.graph.initializer}
    for node in small.graph.node:
        for name in node.output:mapping[name]='logits' if name=='priorfree_logits' else '__priorfree_'+name
    for tensor in small.graph.initializer:tensor.name=mapping[tensor.name]
    for i,node in enumerate(small.graph.node):
        node.name=f'__priorfree_{i}_'+node.name
        for k,name in enumerate(node.input):node.input[k]=mapping.get(name,name)
        for k,name in enumerate(node.output):node.output[k]=mapping.get(name,name)
    graph=onnx.load(PREV/'preprocessing.onnx');graph.graph.node.extend(small.graph.node);graph.graph.initializer.extend(small.graph.initializer)
    del graph.graph.output[:];graph.graph.output.append(helper.make_tensor_value_info('logits',TensorProto.FLOAT,[5]))
    helper.set_model_props(graph,{'parent_sha256':old.SOURCE_SHA,'platform_score':'unknown',
       'method':'Prior-free raw view logodds evidence arbiter. Original1 and non-pair-protected outputs retained. Adjacent alternatives with positive raw odds only.',
       'weights':'Original037 backbone and054 biases unchanged. Two225parameter gate models trained on record-OOF examples.',
       'development':'Repeated room development278to298/880; CI conditionalonfixedrooms, not newblindtest. Source676to647/960, known tradeoff.'})
    onnx.checker.check_model(graph);onnx.save(graph,folder/'model.onnx')
    assert (folder/'model.onnx').stat().st_size<100_000_000
    spec=importlib.util.spec_from_file_location('priorfree_reference_helpers',ROOT/'experiments/protection_arbiter_20261025/export.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);ref.WORK=WORK
    ref.references(head,folder)
    save(folder/'structure_proof.json',dict(parent=old.SOURCE_SHA,source_correct=a['correct'],development=r,
       head_parameters=sum(x.numel() for g in head.gates for x in g.parameters()),new_class_bias=False,original_weights_changed=False))

def publish():
    folder=WORK/'export';check=read(folder/'resource_check.json');assert check['status']=='pass'
    assert read(WORK/'results.json')['passed'];assert read(WORK/'actual_metrics.json')['correct']>=647
    dest=ROOT/'models/prior_free_arbiter_20261025';dest.mkdir(exist_ok=True);path=dest/'01_prior_free_evidence_arbiter.onnx'
    if path.exists():assert sha(path)==check['sha256']
    else:shutil.copy2(folder/'model.onnx',path)
    m=dict(id='prior_free_evidence_arbiter_20261025',name='原始视角对数赔率证据仲裁器',path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
      status='unscored_candidate',platform_score=None,parent_sha256=old.SOURCE_SHA,development=read(WORK/'results.json'),source_metrics=read(WORK/'actual_metrics.json'),resource_check=check,
      report='docs/PROTECTION_ARBITER_20261025.md',risk='Source correct676to647; development rooms reused across many hypotheses, platform improvement unproven.')
    save(path.with_suffix('.json'),m);reg=read(ROOT/'models/registry.json');reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('id')!=m['id']]+[m]
    reg['next_submission']=m;reg['research_status']='prior_free_arbiter_ready_unscored';reg['research_report']=m['report'];save(ROOT/'models/registry.json',reg)
    save(WORK/'published.json',m);print('PUBLISHED',path,flush=True)

if __name__=='__main__':
    if '--publish' in sys.argv:publish()
    else:assemble()
