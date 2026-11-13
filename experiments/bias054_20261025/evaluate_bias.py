"""Small predeclared final-output offsets; never alter frozen054 routing."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'route_attention_20261025'))
from ra_common import *
BWORK=Path('D:/huawei2026-experiments/bias054_20261025')
BASES=dict(selective=dict(path=SOURCE,sha256=SOURCE_SHA),gprime=dict(path=old.SOURCE,sha256=old.SOURCE_SHA))

def data(room,base):
    d=torch.load(PREV/f'{room}.pt',weights_only=True);b,T=old.calibration(room);z=old.policy(d['raw'],b,T)
    if base=='selective':z=prev.transfer(z,old.policy(d['student'],b,T),'low_guard')
    return z,d['labels'],d['rows']

def configs():
    out=[dict(name='control',delta=[0.]*5)]
    for k in range(5):
        for amount in (-.2,-.1,.1,.2):
            delta=[0.]*5;delta[k]=amount;out.append(dict(name=f'class{k}_{amount:+.2f}',delta=delta))
    for amount in (-.05,.05):out.append(dict(name=f'count_tilt_{amount:+.2f}',delta=[amount*(k-2) for k in range(5)]))
    for amount in (-.1,.1):out.append(dict(name=f'zero_one_contrast_{amount:+.2f}',delta=[-amount,amount,0.,0.,0.]))
    return out

def main():
    torch.set_num_threads(4);BWORK.mkdir(parents=True,exist_ok=True);cs=configs()
    save(BWORK/'plan.json',dict(bases={k:dict(path=str(v['path']),sha256=v['sha256']) for k,v in BASES.items()},configs=cs,scope='Append final Add of5FLOAT constants; allinternalbias,normalization,weights,routingremainexactlyunchanged.',selection='Fixed24nonzerooffsetsperbase,no adaptivefivedimensionalgrid. Actual/proxy SAME numericaldelta. Reusedouterrooms aredevelopment,not blind. Require proxypositive,>=2roomsimproved,anyroomloss<=2pp,sourcecorrect>=ownbase,source4personloss<=1; chooselargestproxygain thenworstroom thenfewerchangedproxywindows then smallermaxdelta;tiesprefercompactselective.',source='960preprocessedexactONNXfeatures; priorauditedsource054comparison. Newmodelswillbeauditedonallchangedrawinputs andboundaryprobes.',warning='Priororconditionaldistributionsunknown. Smallcalibrationmayhelpormayhurt. Previous049+0.15onebiasfell048; donotassumemonotonicgain.'))
    results=[]
    for base in ('selective','gprime'):
      assert sha(BASES[base]['path'])==BASES[base]['sha256']
      cache={r:data(r,base) for r in ('a1','c1','g1','e1','actual')}
      for config in cs:
        item=dict(base=base,**config,rooms={});net=0;improve=0;worst=1;changed=0
        for room,(z,y,rows) in cache.items():
            iz=z+z.new_tensor(config['delta']);ix=old.split(room,rows)[2];m=old.metrics(iz,y,z,ix);m['changed']=int((iz[ix].argmax(-1)!=z[ix].argmax(-1)).sum());item['rooms'][room]=m
            if room!='actual':n=m['corrected']-m['broken'];net+=n;improve+=n>0;worst=min(worst,n/m['total']);changed+=m['changed']
        src=item['rooms']['actual'];z,y,rows=cache['actual'];bcm=torch.bincount(y*5+z.argmax(-1),minlength=25).reshape(5,5)
        item.update(proxy_net=net,proxy_changed=changed,improving_rooms=improve,worst_room_delta=worst,source_delta=src['correct']-int(bcm.trace()),source_base_correct=int(bcm.trace()),eligible=net>0 and improve>=2 and worst>=-.02 and src['correct']>=int(bcm.trace()) and src['confusion_matrix'][4][4]>=int(bcm[4,4])-1)
        results.append(item)
    save(BWORK/'results.json',results)
    ordered=sorted(results,key=lambda x:(x['eligible'],x['proxy_net'],x['worst_room_delta'],-x['proxy_changed'],-max(abs(t) for t in x['delta']),x['base']=='selective'),reverse=True)
    for x in ordered[:18]:print('BIAS054',x['base'],x['name'],'proxy',x['proxy_net'],'src',x['source_delta'],'4',x['rooms']['actual']['confusion_matrix'][4][4],'eligible',x['eligible'],{k:v['corrected']-v['broken'] for k,v in x['rooms'].items()},flush=True)
    if ordered[0]['eligible']:save(BWORK/'selection.json',ordered[0])
if __name__=='__main__':main()
