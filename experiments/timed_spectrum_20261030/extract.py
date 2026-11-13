"""Actual-time versus uniform-time versus four-frame-averaged spectral evidence."""
import sys
from pathlib import Path
import numpy as np
import torch
import blosc2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/timed_spectrum_20261030');CACHE=Path('D:/huawei2026-experiments/next053_20261025');RAW=Path('D:/huawei2026-data/labeled')

def signals(a):
 p=a.abs().square();valid=p>1e-8;den=valid.reshape(600,3,32,32).sum(-1).clamp_min(1)
 amp=(.5*torch.log1p(p)*valid).reshape(600,3,32,32).sum(-1)/den
 u=a/torch.sqrt(p+1e-4);phase=[]
 for i,j in ((0,1),(0,2),(1,2)):
  mask=valid[:,i]&valid[:,j];z=(u[:,i]*u[:,j].conj()*mask).reshape(600,32,32).sum(-1)/mask.reshape(600,32,32).sum(-1).clamp_min(1);phase.append(z)
 q=torch.stack(phase,1);return torch.cat([amp,q.real,q.imag],1).reshape(600,288)

def spectrum(x,time):
 # Quadrature weights keep dense packet bursts from being overrepresented.
 gaps=(time[1:]-time[:-1]).clamp_min(0);w=torch.cat([gaps[:1]/2,(gaps[:-1]+gaps[1:])/2,gaps[-1:]/2]);w=w/w.sum().clamp_min(1e-8)
 x=x-(x*w[:,None]).sum(0,keepdim=True)
 edges=torch.tensor([.5,1,2,4,8,16,32,64],device=x.device);f=(edges[:-1,None]+(edges[1:]-edges[:-1])[:,None]*torch.tensor([.125,.375,.625,.875],device=x.device)).flatten()
 angle=2*torch.pi*f[:,None]*time[None,:];c=(torch.cos(angle)*w)@x;s=(torch.sin(angle)*w)@x
 power=(c.square()+s.square()).reshape(7,4,3,3,32).mean(1)
 parts=[power[:,0],power[:,1]+power[:,2]];features=[]
 for v in parts:
  v=torch.log1p(v/(v.mean(0,keepdim=True)+1e-8))
  features.extend([v.mean(1).flatten(),v.std(1,unbiased=False).flatten()])
 return torch.cat(features)

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;WORK.mkdir(parents=True,exist_ok=True);rows=old.read(CACHE/'rows.json');out={k:np.zeros((len(rows),896),np.float32) for k in ('actual','uniform','coarse')};audit=[]
 for n,ident in enumerate(sorted({r['identifier'] for r in rows})):
  a=blosc2.load_array(str(RAW/f'CSI_{ident}.b2'));dt=blosc2.load_array(str(RAW/f'ATI_{ident}.b2')).reshape(-1)
  for k,r in enumerate(rows):
   if r['identifier']!=ident:continue
   raw=torch.as_tensor(np.asarray(a[r['start']:r['stop']]).reshape(600,3,1024),device='cuda');x=signals(raw);interval=torch.tensor(np.array(dt[r['start']:r['start']+599]),device='cuda',dtype=torch.float32).clamp_min(0)/1e6
   time=torch.cat([torch.zeros(1,device='cuda'),interval.cumsum(0)]);assert float(time[-1])>0
   uniform=torch.linspace(0,float(time[-1]),600,device='cuda')
   out['actual'][k]=spectrum(x,time).cpu().numpy();out['uniform'][k]=spectrum(x,uniform).cpu().numpy();out['coarse'][k]=spectrum(x.reshape(150,4,288).mean(1),time.reshape(150,4).mean(1)).cpu().numpy()
   if k==0:
    baseline=spectrum(x,time);reverse=spectrum(x.flip(0),time[-1]-time.flip(0));error=float((baseline-reverse).abs().max());assert error<1e-4,error
    zero=spectrum(torch.zeros_like(x),time);assert torch.isfinite(zero).all();audit.append(dict(reversal_error=error,zero_max=float(zero.abs().max())))
  del a;print('EXTRACT',n+1,'/48',flush=True)
 assert all(np.isfinite(v).all() for v in out.values());np.savez_compressed(WORK/'features.npz',**out)
 old.save(WORK/'protocol.json',dict(rows_sha256=old.sha(CACHE/'rows.json'),code_sha256=old.sha(Path(__file__)),features={k:list(v.shape) for k,v in out.items()},checks=audit,frequency_bands_Hz=[.5,1,2,4,8,16,32,64],method='weighted nonuniform Fourier projections, 4 interior frequencies per octave; normalized band energy, symmetric RX/pair mean/std per32 carrier group',limits='Finite-window spectral descriptors, not independent Doppler/person counts. About3seconds limits low-frequency resolution. Coarse high bands may alias; included as deliberate compression control. Assumes ATI[k] is gap k to k+1 as original preprocessing.'))
if __name__=='__main__':main()
