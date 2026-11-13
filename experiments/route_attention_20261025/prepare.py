"""Capture dynamic/phase/time latents for all12 actual-preprocessed views."""
import gc,time
from ra_common import *
def main():
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(source_sha256=SOURCE_SHA,target=.60,new_architecture='Cross-routeattention on permutation-invariant mean/std of12viewdynamic/phase/temporal latents; bounded sample-dependent residual; current054final2+ unchanged',
      controls=['pooledMLP','crossrouteattention'],training=['balanced','groupDRO'],seeds=[42,2026],steps=[50,150,400],amounts=[.25,.5,1.],
      regularization='weightdecay.05,teacherKL.1,deltaL2.02,8-of12view consistency.02,labelsmoothing.03; shadowallclasses with lowteacherregionweight1 andhigh.25',
      selection='calibrationwhole-recordNLL selects snapshot; actualstepmedianofproxychoices; outerroomlabels notfit. Needpositiveaggregate>=2improvingrooms,no>2pploss;sourcehighcountpreserved;do not claim060from source.',
      papers=['https://arxiv.org/abs/1810.00825','https://arxiv.org/abs/1911.08731']))
    rows=read(PREV/'rows.json');tt=np.load(PREV/'t.npy',mmap_mode='r');ss=np.load(PREV/'s.npy',mmap_mode='r')
    for room in ('a1','c1','g1','e1','actual'):
        path=WORK/f'{room}.pt'
        if path.exists():continue
        model,b=old.build(old.checkpoint(room),full=room=='actual');model.cuda().eval();prior=torch.load(PREV/f'{room}.pt',weights_only=True);captured={};handles=[]
        for name in ('dynamic','phase','temporal'):
            def hook(m,args,out,name=name):captured[name]=out
            handles.append(getattr(model,name).register_forward_hook(hook))
        latents=[];logits=[];started=time.perf_counter()
        with torch.inference_mode():
            for permutation in old.PERMS:
                idx,sign,si=[x.cuda() for x in old.permutation_indices(permutation)];ee=[];zz=[]
                for offset in range(0,960,16):
                    t=torch.from_numpy(np.array(tt[offset:offset+16],copy=True)).cuda().flatten(0,1);s=torch.from_numpy(np.array(ss[offset:offset+16],copy=True)).cuda().flatten(0,1)
                    raw=model.forward_features(t.index_select(1,idx)*sign,s.index_select(1,si));zz.append(raw.cpu().reshape(-1,2,5));ee.append(torch.stack([captured[n] for n in ('dynamic','phase','temporal')],1).cpu().reshape(-1,2,3,256))
                latents.append(torch.cat(ee));logits.append(torch.cat(zz));print('RA_PREP',room,permutation,round(time.perf_counter()-started,1),flush=True)
        error=float((torch.stack(logits,1)-prior['raw']).abs().max());assert error<1e-5,error
        b,T=old.calibration(room);base=prev.transfer(old.policy(prior['raw'],b,T),old.policy(prior['student'],b,T),'low_guard')
        e=torch.stack(latents,1).flatten(1,2);savepath=dict(e=e,stats=moments(e),base=base,aux=auxiliary(prior['raw'],b,T),labels=prior['labels'],rows=rows,checkpoint_sha256=sha(old.checkpoint(room)),base_prediction_error=error)
        torch.save(savepath,path)
        for h in handles:h.remove()
        del model,prior,latents,logits,savepath;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
