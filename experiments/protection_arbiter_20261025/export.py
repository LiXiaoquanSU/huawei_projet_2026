"""Export a qualified reliability arbiter around the unchanged original054."""
import shutil
import blosc2,onnx
from onnx import helper,numpy_helper,TensorProto
from learn import *
from auxiliary import internal_opinions,pooled_aux
SOFT='--soft' in sys.argv
if SOFT:
    WORK=OOF_ROOT/('guard_soft_quarter' if '--max-quarter' in sys.argv else 'guard_soft_half' if '--max-half' in sys.argv else 'guard_soft')
    if ERROR_COST!=2:WORK=WORK/f'cost{ERROR_COST}'
NAMES=[f'__rxg_012___tta_v{t}_{suffix}' for t in (0,1) for suffix in ('cat_8','cat_6')]

def configuration():
    method=read(WORK/'selection.json')['selected'];assert method
    s=read(WORK/'source_selection_actual.json')
    config=s['selected'] if method=='source_selected' else next(c for c in s['candidates'] if c['profile']==method)
    assert config['threshold']<100,'Selected actual gate abstains; do not publish identical baseline.'
    return method,config

class Head(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;self.views=ReceiverViews()
        self.model,b=old.build(old.checkpoint('actual'),full=True);self.register_buffer('bias',b)
        if ALIGNED:self.register_buffer('aux_delta',torch.load(OOF_ROOT/'cal_target_actual.pt',weights_only=True)['delta'])
        cp=final_models('actual',config['profile']);self.arbiters=nn.ModuleList()
        for s in cp['states']:
            gate=Arbiter(cp['profile']);gate.load_state_dict(s['state']);self.arbiters.append(gate)
    def forward(self,t0,s0,t1,s1):
        t,s=self.views(torch.cat([t0,t1]),torch.cat([s0,s1]))
        tasks=self.model.forward_tasks(t,s);raw=tasks['logits'].reshape(1,6,2,5);p=opinions(raw,self.bias,1.)
        if AUX_HEADS:
            extra=internal_opinions(tasks).reshape(6,1,2,6,5).permute(1,0,2,3,4)
            aux=pooled_aux(extra,self.bias,1.)
            if ALIGNED:aux=(aux.clamp_min(1e-12).log()+self.aux_delta).softmax(-1)
            p=torch.cat([p,aux],1)
        regret=torch.stack([gate(p) for gate in self.arbiters]).mean(0)
        out=action(p,regret,self.config['scope'],self.config['threshold'])[0]
        if SOFT:out=((1-self.config['amount'])*p[:,0]+self.config['amount']*out.exp()).clamp_min(1e-12).log()
        return out[0]

def actual_predictions(config):
    d=torch.load(EVID/'actual.pt',weights_only=True);b,T=old.calibration('actual');p=opinions(d['raw'],b,T)
    if AUX_HEADS:p=torch.load(OOF_ROOT/'target_actual.pt',weights_only=True)['probabilities']
    cp=final_models('actual',config['profile']);z,index=apply(p,cp,config);base=old.policy(d['raw'],b,T)
    if SOFT:z=((1-config['amount'])*p[:,0]+config['amount']*z.exp()).clamp_min(1e-12).log()
    metrics=old.metrics(z,d['labels'],base,list(range(len(z))));bm=old.metrics(base,d['labels'],base,list(range(len(z))))
    assert (index!=0).any(),'No changed windows; not a new candidate.'
    save(WORK/'actual_metrics.json',dict(metrics=metrics,baseline=bm,changes=int((index!=0).sum()),note='All48 records trained original teacher. Not independent accuracy.'))
    assert metrics['correct']>=bm['correct']-29,'Source sanity collapse: block publication; inspect calibration/domain transfer.'
    torch.save(dict(logits=z,baseline=base,actions=index,labels=d['labels']),WORK/'actual_predictions.pt')
    return z,index

def assemble():
    torch.set_num_threads(4);method,config=configuration();actual_predictions(config)
    folder=WORK/'export';folder.mkdir(exist_ok=True);head=Head(config).eval()
    example=(torch.zeros(1,963,150),torch.zeros(1,17),torch.zeros(1,963,150),torch.zeros(1,17))
    torch.onnx.export(head,example,str(folder/'head.onnx'),input_names=NAMES,output_names=['arbiter_logits'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(folder/'head.onnx');casts=[];compressed=[]
    for tensor in small.graph.initializer:
        a=numpy_helper.to_array(tensor)
        if a.dtype==np.float32 and a.size>=2048 and np.array_equal(a,a.astype(np.float16).astype(np.float32)):
            name=tensor.name;half=name+'__half_storage';tensor.CopyFrom(numpy_helper.from_array(a.astype(np.float16),name=half))
            casts.append(helper.make_node('Cast',[half],[name],to=TensorProto.FLOAT));compressed.append(name)
    nodes=list(small.graph.node);del small.graph.node[:];small.graph.node.extend(casts+nodes)
    mapping={t.name:'__arbiter_'+t.name for t in small.graph.initializer}
    for n in small.graph.node:
        for v in n.output:mapping[v]='logits' if v=='arbiter_logits' else '__arbiter_'+v
    for t in small.graph.initializer:t.name=mapping[t.name]
    for i,n in enumerate(small.graph.node):
        assert not any(a.type==onnx.AttributeProto.GRAPH for a in n.attribute)
        n.name=f'__arbiter_{i}_'+n.name
        for k,v in enumerate(n.input):n.input[k]=mapping.get(v,v)
        for k,v in enumerate(n.output):n.output[k]=mapping.get(v,v)
    graph=onnx.load(PREV/'preprocessing.onnx');graph.graph.node.extend(small.graph.node);graph.graph.initializer.extend(small.graph.initializer)
    del graph.graph.output[:];graph.graph.output.append(helper.make_tensor_value_info('logits',TensorProto.FLOAT,[5]))
    helper.set_model_props(graph,{'platform_accuracy':'not evaluated','base_sha256':old.SOURCE_SHA,
      'method':'nested-record-OOF conditional protection arbiter '+method,
      'weights':'Original037/044 weights and054 biases unchanged; new relative-regret arbiter and per-aux-head source calibration if aligned.',
      'selection':'Source record-OOF selection; reused whole-room development validation. No hidden-test score claim.',
      'training':'Each gate label generated by a teacher excluding that record and separate calibration records. Outer room excluded from all fitting.'})
    onnx.checker.check_model(graph);onnx.save(graph,folder/'model.onnx');assert (folder/'model.onnx').stat().st_size<100_000_000
    save(folder/'structure_proof.json',dict(method=method,config=config,bytes=(folder/'model.onnx').stat().st_size,
      sha256=sha(folder/'model.onnx'),compressed_tensors=compressed,original_class_bias_changed=False,auxiliary_head_biases=ALIGNED))
    return head,folder

def references(head,folder):
    d=torch.load(WORK/'actual_predictions.pt',weights_only=True);changes=torch.where(d['actions']!=0)[0].tolist()
    cases=list(dict.fromkeys([0,*changes[:3],240,599,959,-1]));prep=make_session(PREV/'preprocessing.onnx')
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
        if i>=0:assert float(np.abs(p(z)-p(d['logits'][i].numpy())).max())<.002
        refs.append(dict(input=name,source_window=i,logits=z.tolist(),probability_error=error))
        print('ARBITER_RAW_CHECK',i,error,flush=True)
    save(folder/'references.json',refs)

def publish():
    folder=WORK/'export';check=read(folder/'resource_check.json');assert check['status']=='pass'
    method,config=configuration();selected=next(x for x in read(WORK/'results.json') if x['method']==method);assert selected['passed']
    dest=ROOT/'models/protection_arbiter_20261025';dest.mkdir(exist_ok=True);path=dest/'01_calibrated_expert_arbiter.onnx'
    if path.exists():assert sha(path)==check['sha256']
    else:shutil.copy2(folder/'model.onnx',path)
    m=dict(id='protection_arbiter_20261025',name='独立分折训练的条件保护仲裁器',path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
      platform_score=None,status='unscored_candidate',parent_sha256=old.SOURCE_SHA,method=method,config=config,development=selected,resource_check=check,
      original_class_bias_changed=False,auxiliary_head_biases=ALIGNED,report='docs/PROTECTION_ARBITER_20261025.md')
    save(path.with_suffix('.json'),m);reg=read(ROOT/'models/registry.json');reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('id')!=m['id']]+[m]
    reg['next_submission']=m;reg['research_status']='conditional_protection_arbiter_ready_unscored';save(ROOT/'models/registry.json',reg);save(WORK/'published.json',m)
    print('ARBITER_PUBLISHED',path,flush=True)

if __name__=='__main__':
    if '--publish' in sys.argv:publish()
    else:
        h,f=assemble();references(h,f)
