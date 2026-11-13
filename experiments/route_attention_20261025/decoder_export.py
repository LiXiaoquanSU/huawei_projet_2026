"""Export selected dual-decoder policy, no new fitted weights."""
from decoder_round import *
from export_batch import NAMES

class DecoderHead(FeatureHead):
    def __init__(self,selection):
        super().__init__('transfer_low_guard');self.selection=selection;self.residuals=nn.ModuleList()
    def forward(self,t0,s0,t1,s1):
        t,s=self.views(torch.cat([t0,t1]),torch.cat([s0,s1]));tasks=self.model.forward_tasks(t,s);raw=tasks['logits'].reshape(1,6,2,5)
        e=torch.stack([self.captured[n] for n in ('static','dynamic','phase','temporal')],1);p,tv=self.student.encode(self.captured['p'],self.captured['t']);se=torch.stack([e[:,0],e[:,1],p,tv],1);sd,so=parts(self.student.decoder,se,s)
        student=((sd+so)*.5).clamp_min(1e-8).log().reshape(1,6,2,5)
        d=dict(raw=raw,student=student,direct=tasks['direct'].reshape(1,6,2,5),ordinal=tasks['ordinal'].reshape(1,6,2,5),student_direct=sd.reshape(1,6,2,5),student_ordinal=so.reshape(1,6,2,5))
        z,base=result(d,self.selection['method'],self.selection['scope'],self.bias,1.)
        self.last_base=base[0];return z[0]

def main():
    import export_routes
    export_routes.WORK=DEC;export_routes.RouteHead=DecoderHead
    export_routes.main()
    import onnx
    from onnx import helper
    path=DEC/'export/model.onnx';m=onnx.load(path)
    helper.set_model_props(m,dict(base_sha256=SOURCE_SHA,base_platform_accuracy='0.54 user reported',platform_accuracy='not evaluated',method='dual supervised categorical/ordinal decoder fusion',selection=json.dumps(read(DEC/'selection.json')),weights='all original and previous quarter-adapter weights unchanged; no newly fit parameters',routing='all old branches decided by unchanged old outputs',preprocessing='exact extracted original054 nodes',storage='exact FP16 storage only, float32 compute'))
    names=[n.name for n in m.graph.node if n.name];assert len(names)==len(set(names)),'Node names must be unique'
    onnx.checker.check_model(m);onnx.save(m,path);proof=read(DEC/'export/structure_proof.json');proof.update(candidate_sha256=sha(path),bytes=path.stat().st_size,method='dual_decoder_fusion',new_parameters=0);save(DEC/'export/structure_proof.json',proof)
    print('DEC_EXPORTED',proof['candidate_sha256'],proof['bytes'],flush=True)
if __name__=='__main__':main()
