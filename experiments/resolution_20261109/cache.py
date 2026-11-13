import importlib.util,gc
from pathlib import Path
import numpy as np
import torch,blosc2
spec=importlib.util.spec_from_file_location('resolution_stage',Path(__file__).with_name('run.py'));stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
base=stage.base;W=stage.W

def transform(h,f,t):
 power=h.abs().square();out=[]
 for u,v in [(0,1),(0,2),(1,2)]:
  a,b=power[:,:,u],power[:,:,v];mask=((a>1e-8)&(b>1e-8)).float();den=(a+b).clamp_min(1e-8);cross=h[:,:,u]*h[:,:,v].conj();coords=torch.stack([2*cross.real/den,2*cross.imag/den,(a-b)/den],2)
  q=(coords*mask[:,:,None]).reshape(len(h),600,3,f,1024//f).sum(-1)/mask.reshape(len(h),600,f,1024//f).sum(-1).clamp_min(1)[:,:,None]
  out.append(q.reshape(len(h),t,600//t,3,f).mean(2).permute(0,2,3,1))
 return torch.cat(out,1)
def main():
 torch.set_num_threads(4);W.mkdir(exist_ok=True);rows=base.load(base.CACHE/'rows.json')
 torch.manual_seed(42);a=stage.Model().eval();b=stage.ResolutionModel().eval();b.load_state_dict(a.state_dict());z=torch.randn(2,9,32,150);s=torch.zeros(2,17)
 with torch.no_grad():err=float((a(z,s)-b(z,s)).abs().max())
 assert err==0;assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
 if (W/'features_complete.json').exists():assert base.load(W/'features_complete.json')['rows']==rows;return
 maps={name:np.lib.format.open_memmap(W/f'{name}.npy',mode='w+',dtype='float32',shape=(len(rows),9,f,t)) for name,f,t in [('time300',32,300),('freq64',64,150)]};prior=np.load(stage.OLD/'joint.npy',mmap_mode='r');maxerr=0
 for ident in sorted({r['identifier'] for r in rows}):
  raw=blosc2.load_array(f'D:/huawei2026-data/labeled/CSI_{ident}.b2');ix=[i for i,r in enumerate(rows) if r['identifier']==ident]
  for lo in range(0,len(ix),4):
   ii=ix[lo:lo+4];h=torch.from_numpy(np.stack([raw[int(rows[i]['start']):int(rows[i]['stop'])].reshape(600,3,1024) for i in ii])).cuda()
   with torch.no_grad():
    t=transform(h,32,300);f=transform(h,64,150);maps['time300'][ii]=t.cpu().numpy();maps['freq64'][ii]=f.cpu().numpy();reduced=t.reshape(len(ii),9,32,150,2).mean(-1).cpu().numpy();maxerr=max(maxerr,float(np.abs(reduced-prior[ii]).max()))
   del h
  print('CACHE',ident,flush=True);del raw;gc.collect()
 for m in maps.values():m.flush();assert np.isfinite(m).all()
 assert maxerr<1e-6,maxerr
 base.save(W/'features_complete.json',dict(rows=rows,baseline_forward_max_error=err,time_reduction_baseline_max_error=maxerr,source='raw CSI; identical carrier validity mask and bounded ratio transform; no labels used for preprocessing'))
 print('CACHE_COMPLETE',maxerr,flush=True)
if __name__=='__main__':main()
