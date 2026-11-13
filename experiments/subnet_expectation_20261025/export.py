"""One backbone, two trained readout masks; export only after the local gate."""
import shutil
import blosc2,onnx
from onnx import helper,numpy_helper,TensorProto
from run import *
sys.path.insert(0,str(ROOT/'experiments/next053_20261025'))
from n53_common import make_session
from common import ReceiverViews
NAMES=[f'__rxg_012___tta_v{t}_{suffix}' for t in (0,1) for suffix in ('cat_8','cat_6')]

class Head(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c;self.model,b=old.build(old.checkpoint('actual'),full=True)
        self.register_buffer('bias',b);self.views=ReceiverViews()
    def forward(self,t0,s0,t1,s1):
        t,s=self.views(torch.cat([t0,t1]),torch.cat([s0,s1]))
        m=self.model;a=t[:,:768].reshape(-1,6,128,150);p=t[:,768:960].reshape(-1,6,32,150)
        e=torch.stack([m.static(a[:,:3]),m.dynamic(a[:,3:]),m.phase(p),m.temporal(a[:,3:],p,t[:,960:])],1)
        raw=decode(m,e,s,(1.,1.,1.,1.)).reshape(1,6,2,5)
        if self.config['kind']=='expected':
            masks,weights=mask_distribution();parts=[decode(m,e,s,mask).exp()*w for mask,w in zip(masks,weights)]
            alt=torch.stack(parts).sum(0).clamp_min(1e-12).log().reshape(1,6,2,5)
        else:
            mask=(1.,1.,1.,0.) if self.config['kind']=='no_time' else (0.,1.,1.,1.)
            alt=decode(m,e,s,mask).reshape(1,6,2,5)
        return policy(raw,alt,self.bias,1.,self.config['scope'],self.config['amount'])[0]

def assemble():
    selected=read(WORK/'selection.json')['selected'];assert selected and selected['robust_pass']
    folder=WORK/'export';folder.mkdir(exist_ok=True);torch.set_num_threads(4)
    head=Head(selected['config']).eval();example=(torch.zeros(1,963,150),torch.zeros(1,17),torch.zeros(1,963,150),torch.zeros(1,17))
    torch.onnx.export(head,example,str(folder/'head.onnx'),input_names=NAMES,output_names=['subnet_logits'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(folder/'head.onnx');casts=[];compressed=[]
    for tensor in small.graph.initializer:
        a=numpy_helper.to_array(tensor)
        if a.dtype==np.float32 and a.size>=2048 and np.array_equal(a,a.astype(np.float16).astype(np.float32)):
            name=tensor.name;half=name+'__half_storage';tensor.CopyFrom(numpy_helper.from_array(a.astype(np.float16),name=half))
            casts.append(helper.make_node('Cast',[half],[name],to=TensorProto.FLOAT));compressed.append(name)
    nodes=list(small.graph.node);del small.graph.node[:];small.graph.node.extend(casts+nodes)
    mapping={t.name:'__subnet_'+t.name for t in small.graph.initializer}
    for n in small.graph.node:
        for v in n.output:mapping[v]='logits' if v=='subnet_logits' else '__subnet_'+v
    for t in small.graph.initializer:t.name=mapping[t.name]
    for i,n in enumerate(small.graph.node):
        assert not any(a.type==onnx.AttributeProto.GRAPH for a in n.attribute)
        n.name=f'__subnet_{i}_'+n.name
        for k,v in enumerate(n.input):n.input[k]=mapping.get(v,v)
        for k,v in enumerate(n.output):n.output[k]=mapping.get(v,v)
    graph=onnx.load(PREV/'preprocessing.onnx');graph.graph.node.extend(small.graph.node);graph.graph.initializer.extend(small.graph.initializer)
    del graph.graph.output[:];graph.graph.output.append(helper.make_tensor_value_info('logits',TensorProto.FLOAT,[5]))
    helper.set_model_props(graph,{'platform_accuracy':'not evaluated','base_sha256':old.SOURCE_SHA,
      'method':selected['key'],'weights':'Original037 weights, original044 half storage, unchanged, no seed310.',
      'architecture':'Shared12viewbackbone, original and route-drop-time count readouts; presence still uses time; average final calibrated probabilities.',
      'selection':'Reused whole-room/whole-record development, all4roomsimprove, not a blind test. Source correct676to657; no hidden-score claim.',
      'bias':'Original054 biases unchanged; no newbias fitting.'})
    onnx.checker.check_model(graph);onnx.save(graph,folder/'model.onnx');assert (folder/'model.onnx').stat().st_size<100_000_000
    save(folder/'structure_proof.json',dict(selected=selected,sha256=sha(folder/'model.onnx'),bytes=(folder/'model.onnx').stat().st_size,
      exact_fp16_storage_tensors=compressed,new_weights=False,new_bias=False))
    print('SUBNET_EXPORTED',(folder/'model.onnx').stat().st_size,flush=True)
    return head,folder

def references(head,folder):
    selected=read(WORK/'selection.json')['selected'];key=selected['key'];d=torch.load(WORK/f'pred_actual_{key}.pt',weights_only=True)
    active=torch.where(d['logits'].argmax(-1)!=d['baseline'].argmax(-1))[0].tolist()
    cases=list(dict.fromkeys([0,*active[:2],240,599,959,-1]));prep=make_session(PREV/'preprocessing.onnx')
    candidate=make_session(folder/'model.onnx');rows=read(PREV/'rows.json');refs=[]
    for i in cases:
        if i<0:feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32));name='zero'
        else:
            r=rows[i];raw=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(raw/f'CSI_{r["identifier"]}.b2'));ati=blosc2.load_array(str(raw/f'ATI_{r["identifier"]}.b2'))
            start,stop=int(r['start']),int(r['stop']);x=np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600,1,3,1024,2);dt=ati[start:stop].astype(np.float32,copy=True);dt[-1]=0
            feed=dict(frames=x,intervals=dt);name=str(folder/f'raw_{i}.npz');np.savez(name,**feed);del csi,ati
        with torch.inference_mode():z=head(*[torch.from_numpy(x) for x in prep.run(None,feed)]).numpy()
        out=candidate.run(None,feed)[0];p=lambda a:np.exp(a-a.max())/np.exp(a-a.max()).sum()
        error=float(np.abs(p(z)-p(out)).max());assert error<.002
        if i>=0:
            reference=d['logits'][i].numpy();assert float(np.abs(p(z)-p(reference)).max())<.002
        refs.append(dict(input=name,source_window=i,logits=z.tolist(),probability_error=error))
        print('SUBNET_RAW_CHECK',i,error,flush=True)
    save(folder/'references.json',refs)

def publish():
    folder=WORK/'export';check=read(folder/'resource_check.json');assert check['status']=='pass'
    selection=read(WORK/'selection.json')['selected'];assert selection['robust_pass']
    dest=ROOT/'models/subnet_expectation_20261025';dest.mkdir(exist_ok=True)
    path=dest/'01_shared_backbone_dual_count_readout.onnx'
    if path.exists():assert sha(path)==check['sha256']
    else:shutil.copy2(folder/'model.onnx',path)
    m=dict(id='dual_count_readout_20261025',name='共享主干双人数读出：保留时序presence',path=path.relative_to(ROOT).as_posix(),
      sha256=sha(path),bytes=path.stat().st_size,platform_score=None,status='unscored_candidate',selection=selection,resource_check=check,
      parent_sha256=old.SOURCE_SHA,new_weights=False,new_bias=False,source='Whole-room development gate passed, hidden platform untested',report='docs/SUBNET_EXPECTATION_20261025.md')
    save(path.with_suffix('.json'),m);reg=read(ROOT/'models/registry.json')
    reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('sha256')!=m['sha256']]+[m]
    reg['next_submission']=m;reg['research_status']='dual_count_readout_ready_unscored';save(ROOT/'models/registry.json',reg)
    save(WORK/'published.json',m);print('SUBNET_PUBLISHED',path,flush=True)

if __name__=='__main__':
    if '--publish' in sys.argv:publish()
    else:
        h,f=assemble();references(h,f)
