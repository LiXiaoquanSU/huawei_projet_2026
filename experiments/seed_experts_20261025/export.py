"""Export only a robust selected expert, sharing exact054 preprocessing."""
import shutil
import blosc2,onnx
from onnx import helper,numpy_helper,TensorProto
from run import *
sys.path.insert(0,str(ROOT/'experiments/next053_20261025'))
from n53_common import make_session
from common import ReceiverViews

NAMES=[f'__rxg_012___tta_v{t}_{suffix}' for t in (0,1) for suffix in ('cat_8','cat_6')]

def selected():
    if '--reduced' in sys.argv:
        s=read(WORK/'reduced_selection.json')['selected'];assert s and (s['robust_pass'] or s['exploratory_eligible'])
        s['exception_note']='Exploratory only: not passed original formal stability gate. Reduced expert views to satisfy runtime; baseline preserved.'
        return s
    selection=read(WORK/'selection.json')
    if '--exploratory' in sys.argv:
        s=next(x for x in read(WORK/'ordinal_summary.json') if x['seed']==310 and x['method']=='adjacent_up')
        assert not s['robust_pass'] and s['pooled_gain']>0 and min(s['room_gains'].values())>=0
        assert s['positive_rooms']>=2 and s['bootstrap']['ci95'][0]>0
        s['exploratory_only']=True
        s['exception_note']='Failed predeclared +2pp gate (+1.59pp). Saved as exploratory artifact, not accepted upgrade; baseline/registry next_submission unchanged.'
        save(WORK/'exploratory_selection.json',s)
        return s
    s=selection['selected'];assert s and s['robust_pass'];return s

class Head(torch.nn.Module):
    def __init__(self,seed,method,group=None):
        super().__init__();self.method=method;self.views=ReceiverViews();self.group=group
        self.base,bb=old.build(old.checkpoint('actual'),full=True)
        self.expert,_=old.build(PACKAGE/f'weights/checkpoints/base037_full_s{seed}.pt')
        suffix='_'+group if group else ''
        cal=read(WORK/f'cal_actual_s{seed}{suffix}.json');self.register_buffer('base_bias',bb)
        self.register_buffer('expert_bias',torch.tensor(cal['bias']));self.T=cal['temperature']
        self.permutations=cal.get('permutations',list(range(6)))
        self.register_buffer('expert_indices',torch.tensor([i for p in self.permutations for i in (2*p,2*p+1)]))
    def forward(self,t0,s0,t1,s1):
        t,s=self.views(torch.cat([t0,t1]),torch.cat([s0,s1]))
        raw=self.base.forward_features(t,s).reshape(1,6,2,5)
        second=self.expert.forward_features(t.index_select(0,self.expert_indices),s.index_select(0,self.expert_indices)).reshape(1,len(self.permutations),2,5)
        base=old.policy(raw,self.base_bias)
        return combine(base,second,self.expert_bias,self.T,self.method,raw,self.base_bias)[0]

def assemble():
    choice=selected()
    seed=choice['seed'];method=choice['method'];folder=WORK/('export_'+choice['group'] if choice.get('group') else 'export');folder.mkdir(exist_ok=True)
    head=Head(seed,method,choice.get('group')).eval();torch.set_num_threads(4)
    examples=(torch.zeros(1,963,150),torch.zeros(1,17),torch.zeros(1,963,150),torch.zeros(1,17))
    torch.onnx.export(head,examples,str(folder/'head.onnx'),input_names=NAMES,output_names=['seed_logits'],
                      opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(folder/'head.onnx');casts=[];compressed=[]
    for tensor in small.graph.initializer:
        a=numpy_helper.to_array(tensor)
        if a.dtype==np.float32 and a.size>=2048 and np.array_equal(a,a.astype(np.float16).astype(np.float32)):
            name=tensor.name;new=name+'__half_storage';tensor.CopyFrom(numpy_helper.from_array(a.astype(np.float16),name=new))
            casts.append(helper.make_node('Cast',[new],[name],to=TensorProto.FLOAT));compressed.append(name)
    nodes=list(small.graph.node);del small.graph.node[:];small.graph.node.extend(casts+nodes)
    mapping={t.name:'__sx_'+t.name for t in small.graph.initializer}
    for n in small.graph.node:
        for value in n.output:mapping[value]='logits' if value=='seed_logits' else '__sx_'+value
    for t in small.graph.initializer:t.name=mapping[t.name]
    for node_index,n in enumerate(small.graph.node):
        assert not any(a.type==onnx.AttributeProto.GRAPH for a in n.attribute)
        n.name=f'__sx_{node_index}_'+n.name
        for i,v in enumerate(n.input):n.input[i]=mapping.get(v,v)
        for i,v in enumerate(n.output):n.output[i]=mapping.get(v,v)
    graph=onnx.load(PREV/'preprocessing.onnx');graph.graph.node.extend(small.graph.node);graph.graph.initializer.extend(small.graph.initializer)
    del graph.graph.output[:];graph.graph.output.append(helper.make_tensor_value_info('logits',TensorProto.FLOAT,[5]))
    helper.set_model_props(graph,{'method':f'guarded_seed_{seed}_{method}', 'platform_accuracy':'not evaluated',
       'control_sha256':old.SOURCE_SHA,'second_checkpoint_sha256':sha(PACKAGE/f'weights/checkpoints/base037_full_s{seed}.pt'),
       'selection':'reused held-room development and whole-recordbootstrap; no hidden truth',
       'inference':f'two independent weight sets, sharedexact054 preprocessing, base12views expert{len(head.permutations)*2}views; conditions retained in graph',
       'selection_status':'exploratory, failed formal gate' if choice['exploratory_only'] else 'local gate passed, platform unknown'})
    onnx.checker.check_model(graph);onnx.save(graph,folder/'model.onnx')
    assert (folder/'model.onnx').stat().st_size<100_000_000
    save(folder/'proof.json',dict(selected=choice,compressed_exact_half_tensors=compressed,sha256=sha(folder/'model.onnx'),
                                bytes=(folder/'model.onnx').stat().st_size))
    return head,folder

def references(head,folder):
    rows=read(PREV/'rows.json');prep=make_session(PREV/'preprocessing.onnx')
    candidate=make_session(folder/'model.onnx');refs=[]
    # Include real windows where this particular reduced/full ordinal repair is active.
    choice=selected();suffix='_'+choice['group'] if choice.get('group') else ''
    diagnostic=torch.load(WORK/f'pred_actual_s{choice["seed"]}_{choice["method"]}{suffix}.pt',weights_only=True)
    active=torch.where(diagnostic['logits'].argmax(-1)!=diagnostic['baseline'].argmax(-1))[0].tolist()
    cases=list(dict.fromkeys([0,*active[:2],240,599,959,-1]))
    for i in cases:
        if i<0:feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32));name='zero'
        else:
            r=rows[i];raw=Path('D:/huawei2026-data/labeled')
            csi=blosc2.load_array(str(raw/f'CSI_{r["identifier"]}.b2'));ati=blosc2.load_array(str(raw/f'ATI_{r["identifier"]}.b2'))
            from utils import window
            x,dt=window(csi,ati,r);feed=dict(frames=x,intervals=dt)
            name=str(folder/f'raw_{i}.npz');np.savez(name,**feed);del csi,ati
        with torch.inference_mode():z=head(*[torch.from_numpy(x) for x in prep.run(None,feed)]).numpy()
        out=candidate.run(None,feed)[0];p=lambda a:np.exp(a-a.max())/np.exp(a-a.max()).sum()
        error=float(np.abs(p(z)-p(out)).max());assert error<.002,(i,error)
        refs.append(dict(input=name,logits=z.tolist(),probability_error=error,source_window=i))
        print('EXPORT_RAW_CHECK',i,error,flush=True)
    save(folder/'references.json',refs)

def publish():
    s=selected();folder=WORK/('export_'+s['group'] if s.get('group') else 'export');check=read(folder/'resource_check.json');assert check['status']=='pass'
    dest=ROOT/'models/seed_experts_20261025'
    prefix='02_exploratory' if s.get('exploratory_only') else '01_guarded'
    suffix='_'+s['group'] if s.get('group') else ''
    dest.mkdir(exist_ok=True);name=f'{prefix}_s{s["seed"]}_{s["method"]}{suffix}.onnx';path=dest/name
    if path.exists():assert sha(path)==check['sha256']
    else:shutil.copy2(folder/'model.onnx',path)
    meta=dict(path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=sha(path),bytes=path.stat().st_size,
              status='unscored_exploratory' if s.get('exploratory_only') else 'unscored_candidate',platform_score=None,selection=s,resource_check=check,
              note='Not a claimed060 model. Two seeds, not simple averaging; reused-room proxy evidence only.')
    save(path.with_suffix('.json'),meta)
    registry=read(ROOT/'models/registry.json');registry['research_candidates']=[x for x in registry.get('research_candidates',[]) if x.get('sha256')!=meta['sha256']]+[meta]
    if not s.get('exploratory_only'):registry['next_submission']=meta
    save(ROOT/'models/registry.json',registry)
    save(WORK/'published.json',meta);print('PUBLISHED',path,flush=True)

if __name__=='__main__':
    if '--publish' in sys.argv:publish()
    elif '--references-only' in sys.argv:
        choice=selected();references(Head(choice['seed'],choice['method'],choice.get('group')).eval(),WORK/('export_'+choice['group'] if choice.get('group') else 'export'))
    else:
        h,f=assemble();references(h,f)
