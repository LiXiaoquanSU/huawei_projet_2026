"""Standalone raw CSI -> newly trained joint evidence chain; embedded ONNX."""
import shutil
import blosc2,onnx,onnxruntime as ort
from onnx import helper,TensorProto
from train_chain import *
LAST='--last' in sys.argv

def session(path):
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])

class Head(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=EvidenceChain('full');selection=old.read(WORK/'selection.json');cp=torch.load(WORK/f'actual_full_s{selection["delivery_seed"]}'/('last.pt' if LAST else 'final.pt'),weights_only=True);self.model.load_state_dict(cp['model_state'])
    def forward(self,t0,s0,t1,s1):
        p=self.model(torch.cat([t0,t1]),torch.cat([s0,s1])).exp().mean(0)
        return p.clamp_min(1e-12).log()

def main():
    torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);folder=WORK/('export_last' if LAST else 'export');folder.mkdir(exist_ok=True)
    if '--publish' in sys.argv:
        check=old.read(folder/'resource_check.json');assert check['status']=='pass';dest=ROOT/'models/evidence_large_20261028';dest.mkdir(exist_ok=True);path=dest/('02_large_relation_chain_epoch15_seed42.onnx' if LAST else '01_large_relation_chain_seed42.onnx');shutil.copy2(folder/'model.onnx',path)
        selection=old.read(WORK/'selection.json');meta=dict(id='evidence_large_20261028',path=path.relative_to(ROOT).as_posix(),sha256=old.sha(path),bytes=path.stat().st_size,platform_score=None,status='unscored_research_model',recommended_over054=False,training=selection,resource_check=check,report='docs/EVIDENCE_LARGE_20261028.md',note='New end-to-end architecture, fixed primary seed42; research artifact, not a claim of higher platform score')
        if LAST:meta.update(id='evidence_large_last15_20261028',training_epoch=15,note='Fixed15-epoch research control, not selected by target scores; no claim of higher platform score')
        old.save(path.with_suffix('.json'),meta);old.save(WORK/('published_last.json' if LAST else 'published.json'),meta)
        reg=old.read(ROOT/'models/registry.json');reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('id')!=meta['id']]+[meta];reg.update(updated_date='2026-10-28',research_status='large_relation_chain_trained',research_report=meta['report'],next_submission=None);old.save(ROOT/'models/registry.json',reg);print('PUBLISHED',str(path),meta['bytes'],flush=True);return
    head=Head().eval();names=[f'__rxg_012___tta_v{v}_{suffix}' for v in (0,1) for suffix in ('cat_8','cat_6')]
    examples=(torch.zeros(1,963,150),torch.zeros(1,17),torch.zeros(1,963,150),torch.zeros(1,17))
    torch.onnx.export(head,examples,str(folder/'head.onnx'),input_names=names,output_names=['chain_output'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(folder/'head.onnx');mapping={v.name:'__chain_'+v.name for v in small.graph.initializer}
    for node in small.graph.node:
        for v in node.output:mapping[v]='logits' if v=='chain_output' else '__chain_'+v
    for v in small.graph.initializer:v.name=mapping[v.name]
    for i,node in enumerate(small.graph.node):
        node.name=f'__chain_{i}_'+node.name
        for j,v in enumerate(node.input):node.input[j]=mapping.get(v,v)
        for j,v in enumerate(node.output):node.output[j]=mapping.get(v,v)
    graph=onnx.load(CACHE/'preprocessing.onnx');graph.graph.node.extend(small.graph.node);graph.graph.initializer.extend(small.graph.initializer);del graph.graph.output[:];graph.graph.output.append(helper.make_tensor_value_info('logits',TensorProto.FLOAT,[5]))
    helper.set_model_props(graph,{'architecture':'Multiscale directed receiver graph with contextual temporal attention','weights':'New seed42, all48 labeled records; no old learned weights','training_epoch':str(15 if LAST else old.read(WORK/'selection.json')['epochs']),'platform_score':'unknown','decision':'five-class joint probability, average forward and reverse time; no fixed class bias or protection rules','limits':'research model; no evidence yet of higher platform score'})
    onnx.checker.check_model(graph);onnx.save(graph,folder/'model.onnx');del graph,small
    prep=session(CACHE/'preprocessing.onnx');candidate=session(folder/'model.onnx');d=torch.load(WORK/'actual_full_s42/predictions.pt',weights_only=True);rows=d['rows'];refs=[]
    indices=[next(i for i,r in enumerate(rows) if r['headcount']==label) for label in range(5)]+[599,959,-1]
    soft=lambda x:np.exp(x-x.max())/np.exp(x-x.max()).sum()
    for i in indices:
        if i<0:feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32));name='zero'
        else:
            r=rows[i];root=Path('D:/huawei2026-data/labeled');a=blosc2.load_array(str(root/f'CSI_{r["identifier"]}.b2'));ati=blosc2.load_array(str(root/f'ATI_{r["identifier"]}.b2')).reshape(-1)
            frames=np.ascontiguousarray(a[r['start']:r['stop']]).view(np.float32).reshape(600,1,3,1024,2);dt=ati[r['start']:r['stop']].astype(np.float32,copy=True);dt[-1]=0;feed=dict(frames=frames,intervals=dt);name=str(folder/f'raw_{i}.npz');np.savez(name,**feed);del a,ati
        with torch.inference_mode():z=head(*[torch.from_numpy(v) for v in prep.run(None,feed)]).numpy()
        out=candidate.run(None,feed)[0];error=float(np.max(np.abs(soft(z)-soft(out))));assert error<.002
        if i>=0:assert float(np.max(np.abs(soft(z)-soft(d['last_logits' if LAST else 'logits'][i].numpy()))))<.002
        assert np.isfinite(out).all();refs.append(dict(input=name,source_window=i,logits=z.tolist(),probability_error=error));print('RAW_CHECK',i,error,flush=True)
    old.save(folder/'references.json',refs)
if __name__=='__main__':main()
