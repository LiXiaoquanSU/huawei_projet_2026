"""Input-level receiver-equivariant directed phase graph, not output-view attention."""
from run import *
from model import AttentionBlock

class Encoder(nn.Module):
    def __init__(self,cin,widths,output):
        super().__init__();blocks=[]
        for cout in widths:
            blocks.extend([nn.Conv2d(cin,cout,3,2,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU()]);cin=cout
        self.body=nn.Sequential(*blocks);self.project=nn.Conv1d(2*widths[-1],output,1)
    def forward(self,x):
        x=self.body(x);mean=x.mean(2);std=(x.sub(mean[:,:,None]).square().mean(2)+1e-5).sqrt()
        return self.project(torch.cat([mean,std],1)).transpose(1,2)

class ReceiverGraph(nn.Module):
    def __init__(self):
        super().__init__();self.nodes=Encoder(2,[32,64,96],96);self.edges=Encoder(4,[16,32,48],48)
        source=[0,0,1,1,2,2];dest=[1,2,0,2,0,1]
        self.register_buffer('source',torch.tensor(source));self.register_buffer('dest',torch.tensor(dest))
        self.register_buffer('cosine',torch.tensor([0,2,0,4,2,4]));self.register_buffer('sine',torch.tensor([1,3,1,5,3,5]))
        self.register_buffer('sign',torch.tensor([1,1,-1,1,-1,-1]).float().reshape(1,6,1,1))
        inc=torch.zeros(3,6)
        for j,i in enumerate(source):inc[i,j]=.5
        self.register_buffer('incidence',inc)
        self.node_meta=nn.Sequential(nn.Linear(100,96),nn.LayerNorm(96),nn.SiLU())
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(240,96),nn.SiLU(),nn.Linear(96,96)) for _ in range(2)])
        self.norms=nn.ModuleList([nn.LayerNorm(96) for _ in range(2)])
        self.project=nn.Linear(336,192);self.clock=nn.Linear(3,192)
        self.temporal=nn.Sequential(AttentionBlock(192),AttentionBlock(192))
        self.head=nn.Sequential(nn.Linear(576+9,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.25),nn.Linear(256,5))
    def forward(self,t,s):
        B=t.shape[0];a=t[:,:768].reshape(-1,6,128,150);p=t[:,768:960].reshape(-1,6,32,150)
        node=torch.stack([a[:,:3],a[:,3:]],2).reshape(-1,2,128,150)
        h=self.nodes(node).reshape(B,3,19,96)
        meta=s[:,:12].reshape(-1,4,3).transpose(1,2)[:,:,None].expand(-1,-1,19,-1)
        h=self.node_meta(torch.cat([h,meta],-1))
        dyn=a[:,3:].reshape(-1,3,32,4,150).mean(3)
        edge=torch.stack([p.index_select(1,self.cosine),p.index_select(1,self.sine)*self.sign,
                          dyn.index_select(1,self.source),dyn.index_select(1,self.dest)],2).reshape(-1,4,32,150)
        ep=self.edges(edge).reshape(B,6,19,48)
        for message,norm in zip(self.messages,self.norms):
            m=message(torch.cat([h.index_select(1,self.source),h.index_select(1,self.dest),ep],-1))
            h=norm(h+torch.einsum('ij,bjtc->bitc',self.incidence,m))
        mean=h.mean(1);std=(h.sub(mean[:,None]).square().mean(1)+1e-5).sqrt()
        tokens=self.project(torch.cat([mean,std,h.amax(1),ep.mean(1)],-1))
        time=t[:,960:,::8].transpose(1,2)
        clock=(time[:,:,:1]-time[:,:1,:1])/(time[:,-1:,:1]-time[:,:1,:1]).clamp_min(.01)
        tokens=tokens+self.clock(torch.cat([clock,time[:,:,1:]],-1));tokens=self.temporal(tokens)
        mean=tokens.mean(1);std=(tokens.sub(mean[:,None]).square().mean(1)+1e-5).sqrt()
        # Four Rx-symmetric summary groups + five global timing summaries.
        invariant_summary=torch.cat([s[:,:12].reshape(-1,4,3).mean(-1),s[:,12:17]],-1)
        return self.head(torch.cat([mean,std,tokens.amax(1),invariant_summary],-1))

def fallback(raw,b,T):
    q=(raw.exp().mean(2).clamp_min(1e-12).log()/T+b).exp()
    return q.mean(1).clamp_min(1e-12).log()

def hybrid(raw,student,b,T,amount):
    original=fallback(raw,b,T)
    p=(1-amount)*original.softmax(-1)+amount*student.softmax(-1)
    proposed=p.clamp_min(1e-12).log()+original.logsumexp(-1,keepdim=True)
    return old.finish(raw,b,T,proposed)

def student_scores(model,t,s,batch=32):
    zz=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(t),batch):
            out=model(t[start:start+batch].flatten(0,1),s[start:start+batch].flatten(0,1)).reshape(-1,2,5)
            zz.append(out.softmax(-1).mean(1).clamp_min(1e-12).log().cpu())
    return torch.cat(zz)

def invariance(model,t,s):
    model.eval();errors=[]
    with torch.no_grad():
        z=model(t,s).softmax(-1)
        for p in old.PERMS:
            idx,sign,si=[v.to(t.device) for v in old.permutation_indices(p)]
            other=model(t.index_select(1,idx)*sign,s.index_select(1,si)).softmax(-1)
            errors.append(float((z-other).abs().max()))
    assert max(errors)<1e-5,errors
    return max(errors)
