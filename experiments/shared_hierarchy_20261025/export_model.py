"""Export standalone new architecture, with exact audited preprocessing."""
import shutil
import blosc2,onnx
from onnx import helper,TensorProto
from train import *
HARD='--hard' in sys.argv

def decision(z):
    if not HARD:return z
    empty=z[...,:1].exp()>=.5
    conditional=z[...,1:].softmax(-1)
    return torch.cat([torch.where(empty,torch.ones_like(z[...,:1]),torch.full_like(z[...,:1],1e-12)),torch.where(empty,torch.full_like(conditional,1e-12),conditional)],-1).clamp_min(1e-12).log()

class Head(nn.Module):
    def __init__(self):
        super().__init__();selection=read(WORK/'selection.json');self.models=nn.ModuleList()
        for seed in [selection['delivery_seed']]:
            cp=torch.load(WORK/f'actual_{selection["variant"]}_s{seed}'/'final.pt',weights_only=True)
            m=SharedHierarchy();m.load_state_dict(cp['model_state']);self.models.append(m)
    def forward(self,t0,s0,t1,s1):
        t=torch.cat([t0,t1]);s=torch.cat([s0,s1])
        p=torch.stack([m(t,s).exp() for m in self.models]).mean((0,1))
        return decision(p.clamp_min(1e-12).log())

def assemble():
    torch.set_num_threads(4);folder=WORK/('export_hard' if HARD else 'export');folder.mkdir(exist_ok=True);head=Head().eval()
    names=[f'__rxg_012___tta_v{t}_{suffix}' for t in (0,1) for suffix in ('cat_8','cat_6')]
    example=(torch.zeros(1,963,150),torch.zeros(1,17),torch.zeros(1,963,150),torch.zeros(1,17))
    torch.onnx.export(head,example,str(folder/'head.onnx'),input_names=names,output_names=['hierarchy_logits'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(folder/'head.onnx');mapping={t.name:'__hierarchy_'+t.name for t in small.graph.initializer}
    for n in small.graph.node:
        for v in n.output:mapping[v]='logits' if v=='hierarchy_logits' else '__hierarchy_'+v
    for t in small.graph.initializer:t.name=mapping[t.name]
    for i,n in enumerate(small.graph.node):
        n.name=f'__hierarchy_{i}_'+n.name
        for k,v in enumerate(n.input):n.input[k]=mapping.get(v,v)
        for k,v in enumerate(n.output):n.output[k]=mapping.get(v,v)
    graph=onnx.load(PREV/'preprocessing.onnx');graph.graph.node.extend(small.graph.node);graph.graph.initializer.extend(small.graph.initializer)
    del graph.graph.output[:];graph.graph.output.append(helper.make_tensor_value_info('logits',TensorProto.FLOAT,[5]))
    helper.set_model_props(graph,{'platform_score':'unknown','architecture':'Shared receiver and phase-pair multiscale hierarchical count, source-selected single seed, two time directions','weights':'Newly trained on all48 labeled records, no original037 learned weights, no external or U data','selection':'Source calibration selects epochs and delivery seed; reused leave-room development chooses variant. No blind-test accuracy claim.'})
    prop=graph.metadata_props.add();prop.key='decision';prop.value='Fixed0.5 presence then conditional count; decision scores, not calibrated probabilities' if HARD else 'Joint class probability argmax'
    onnx.checker.check_model(graph);onnx.save(graph,folder/'model.onnx')
    selection=read(WORK/'selection.json');ds=[torch.load(WORK/f'actual_{selection["variant"]}_s{seed}'/'predictions.pt',weights_only=True) for seed in [selection['delivery_seed']]]
    logits=decision(torch.stack([d['logits'].exp() for d in ds]).mean(0).clamp_min(1e-12).log());d=ds[0]
    ref=torch.load(EVID/'actual.pt',weights_only=True);b,T=old.calibration('actual');base=old.policy(ref['raw'],b,T)
    metrics=old.metrics(logits,d['labels'],base,list(range(960)));save(WORK/('hard_actual_metrics.json' if HARD else 'actual_metrics.json'),metrics)
    torch.save(dict(logits=logits,labels=d['labels'],baseline=base),WORK/('hard_actual_predictions.pt' if HARD else 'actual_predictions.pt'))
    indices=[next(i for i,r in enumerate(d['rows']) if int(r['headcount'])==label) for label in range(5)]+[599,959,-1]
    prep=make_session(PREV/'preprocessing.onnx');candidate=make_session(folder/'model.onnx');refs=[]
    for i in dict.fromkeys(indices):
        if i<0:feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32));name='zero'
        else:
            r=d['rows'][i];raw=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(raw/f'CSI_{r["identifier"]}.b2'));ati=blosc2.load_array(str(raw/f'ATI_{r["identifier"]}.b2'));start,stop=int(r['start']),int(r['stop'])
            frames=np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600,1,3,1024,2);dt=ati[start:stop].astype(np.float32,copy=True);dt[-1]=0;feed=dict(frames=frames,intervals=dt);name=str(folder/f'raw_{i}.npz');np.savez(name,**feed);del csi,ati
        with torch.inference_mode():z=head(*[torch.from_numpy(v) for v in prep.run(None,feed)]).numpy()
        out=candidate.run(None,feed)[0];prob=lambda x:np.exp(x-x.max())/np.exp(x-x.max()).sum();err=float(np.abs(prob(z)-prob(out)).max());assert err<.002
        if i>=0:assert float(np.abs(prob(z)-prob(logits[i].numpy())).max())<.002
        refs.append(dict(input=name,source_window=i,logits=z.tolist(),probability_error=err));print('RAW_CHECK',i,err,flush=True)
    save(folder/'references.json',refs);print('SOURCE',metrics,flush=True)

def publish():
    folder=WORK/('export_hard' if HARD else 'export');check=read(folder/'resource_check.json');assert check['status']=='pass';selection=read(WORK/'selection.json')
    dest=ROOT/'models/shared_hierarchy_20261025';dest.mkdir(exist_ok=True);path=dest/(f'02_hard_hierarchy_seed{selection["delivery_seed"]}.onnx' if HARD else f'01_shared_hierarchy_seed{selection["delivery_seed"]}.onnx');shutil.copy2(folder/'model.onnx',path)
    dev=selection['development'];source=read(WORK/'actual_metrics.json')
    improved=dev['net']>0 and dev['ci95'][0]>0 and source['correct']>=676 and all((m['correct']-m['baseline_correct'])/m['total']>=-.02 for m in dev['rooms'].values())
    metadata=dict(id='shared_hierarchy_20261025',name='共享天线多尺度分层人数网络单种子',path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size,status='unscored_exploratory_architecture',platform_score=None,development=selection,source_metrics=read(WORK/'actual_metrics.json'),resource_check=check,recommended_over054=improved,report='docs/SHARED_HIERARCHY_20261025.md')
    if HARD:
        metadata.update(id='hard_hierarchy_20261025',name='固定0.5阈值分层判决实验',development=read(WORK/'hard_results.json'),source_metrics=read(WORK/'hard_actual_metrics.json'),recommended_over054=False,note='Post-hoc fixed-threshold diagnostic; not calibrated probabilities. No threshold sweep or new training.')
    save(path.with_suffix('.json'),metadata);reg=read(ROOT/'models/registry.json');reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('id')!=metadata['id']]+[metadata];reg['research_status']='shared_hierarchy_trained_evaluated';reg['research_report']=metadata['report'];reg['next_submission']=metadata if metadata['recommended_over054'] else None;save(ROOT/'models/registry.json',reg);save(WORK/('hard_published.json' if HARD else 'published.json'),metadata)
    print('PUBLISHED',path,'local_evidence_to_replace054',improved,flush=True)

if __name__=='__main__':
    if '--publish' in sys.argv:publish()
    else:assemble()
