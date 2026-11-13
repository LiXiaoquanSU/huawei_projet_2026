"""Second-stage limited2D followup; explicitly adaptive development, not blind."""
from evaluate_bias import *
def main():
    cs=[dict(name=f'zero_minus010_two_plus{v:.2f}',delta=[-.1,0.,v,0.,0.]) for v in (.1,.2)]
    save(BWORK/'followup_plan.json',dict(reason='Firststage24fixed directionsperbase:2person+.2haslargeproxygainbutsource-1;0person-.1hassourcegainandproxypositiveinone room. Testcomplementarityofthese2directions,notrelaxeligibility.',configs=cs,bases=['selective','gprime'],selection='Samefirststagecriteria. Fouradditionalcomparisons, proposedafterviewingdevelopmentresults; noindependentconfirmation.'))
    out=[]
    for base in ('selective','gprime'):
      for c in cs:
        item=dict(base=base,**c,rooms={},stage='adaptive_followup');net=0;improve=0;worst=1;changed=0
        for room in ('a1','c1','g1','e1','actual'):
            z,y,rows=data(room,base);new=z+z.new_tensor(c['delta']);ix=old.split(room,rows)[2];m=old.metrics(new,y,z,ix);m['changed']=int((new[ix].argmax(-1)!=z[ix].argmax(-1)).sum());item['rooms'][room]=m
            if room!='actual':n=m['corrected']-m['broken'];net+=n;improve+=n>0;worst=min(worst,n/m['total']);changed+=m['changed']
            else:bcm=torch.bincount(y*5+z.argmax(-1),minlength=25).reshape(5,5)
        src=item['rooms']['actual'];item.update(proxy_net=net,proxy_changed=changed,improving_rooms=improve,worst_room_delta=worst,source_delta=src['correct']-int(bcm.trace()),source_base_correct=int(bcm.trace()),eligible=net>0 and improve>=2 and worst>=-.02 and src['correct']>=int(bcm.trace()) and src['confusion_matrix'][4][4]>=int(bcm[4,4])-1);out.append(item)
    save(BWORK/'followup_results.json',out)
    out.sort(key=lambda x:(x['eligible'],x['proxy_net'],x['worst_room_delta'],-x['proxy_changed'],-max(abs(t) for t in x['delta']),x['base']=='selective'),reverse=True)
    for x in out:print('BIAS_COMBO',x['base'],x['delta'],'proxy',x['proxy_net'],'source',x['source_delta'],'eligible',x['eligible'],{k:v['corrected']-v['broken'] for k,v in x['rooms'].items()},flush=True)
    if out[0]['eligible']:save(BWORK/'selection.json',out[0])
if __name__=='__main__':main()
