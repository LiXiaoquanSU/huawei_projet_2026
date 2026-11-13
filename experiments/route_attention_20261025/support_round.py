"""Predeclared label-free support gate on frozen residual controls."""
from ra_common import *
ROUND=Path('D:/huawei2026-experiments/support_gate_20261025')

def make_support(x,rows,fit,cal):
    records=sorted({rows[i]['identifier'] for i in fit});ids=[rows[i]['identifier'] for i in fit]
    proto=torch.stack([x[[i for i in fit if rows[i]['identifier']==r]].mean(0) for r in records])
    center=proto.mean(0);scale=proto.std(0,unbiased=False).clamp_min(.05)
    p=(proto-center)/scale;xx=(x-center)/scale
    distances=(xx[:,None,:]-p[None]).square().mean(-1).sqrt()
    if cal:caldist=distances[cal].amin(-1);method='calibration_records95quantile'
    else:
        mask=torch.tensor([[rows[i]['identifier']==r for r in records] for i in fit]);v=distances[fit].masked_fill(mask,float('inf'));caldist=v.amin(-1);method='fit_leave_record_out95quantile'
    threshold=torch.quantile(caldist,.95).clamp_min(1e-6);score=distances.amin(-1)
    return score<=threshold,dict(center=center,scale=scale,prototypes=proto,threshold=threshold,record_ids=records,threshold_method=method),score

def correction(d,room,family):
    values=[]
    for seed in (42,2026):
        folder=WORK/('binary_fits' if family=='binary' else 'fits')/f"{'mlp' if family=='binary' else family}_balanced_{seed}_{room}"
        step=read(folder/'complete.json')['selected_step'];v=torch.load(folder/('shifts.pt' if family=='binary' else 'deltas.pt'),weights_only=True)[step];values.append(v)
    base=d['base'];v=torch.stack(values)
    if family=='binary':
        from binary_routes import binary_deploy
        proposal=binary_deploy(base,v.mean(0),.5);individual=[binary_deploy(base,z,.5).argmax(-1) for z in v]
    else:proposal=deploy(base,v.mean(0),.5);individual=[deploy(base,z,.5).argmax(-1) for z in v]
    return proposal,individual[0]==individual[1]

def main():
    torch.set_num_threads(4);ROUND.mkdir(parents=True,exist_ok=True)
    plan=dict(base_sha256=SOURCE_SHA,pretrained_heads='Previous pool/attention balanced and binaryMLP balanced, seeds42+2026, unchanged calibration snapshots; no further head fitting.',families=['pool','attention','binary'],support_features=['route_stats','relative34'],amount=.5,seed_agreement_required=True,support='Nearest whole-fit-record prototype in diagonal standardized feature space; threshold95% of held calibration record windows, no labels; actual threshold fromleave-record-out fit windows.',selection='Six fixed configurations. Same eligibility totalnetpositive,>=2improvingrooms,anyroomloss<=2pp,source>=679; prefer positive eachseed and stable architecture; no threshold grid.',caveat='All outer rooms reused development; no independent blind test. Full-source prototypes see source records; source diagnostic remains optimistic.')
    save(ROUND/'plan.json',plan);results=[]
    for feature in ('route_stats','relative34'):
      summaries={family:dict(family=family,feature=feature,amount=.5,rooms={}) for family in ('pool','attention','binary')}
      for room in ('a1','c1','g1','e1','actual'):
        d=torch.load(WORK/f'{room}.pt',weights_only=True);x=d['stats'].flatten(1) if feature=='route_stats' else torch.load(WORK/f'binary_x_{room}.pt',weights_only=True)
        fit,cal,test=old.split(room,d['rows']);gate,state,scores=make_support(x,d['rows'],fit,cal);torch.save(state,ROUND/f'support_{feature}_{room}.pt')
        for family in summaries:
            proposal,agree=correction(d,room,family);mask=gate&agree&(d['base'].argmax(-1)<2);z=torch.where(mask[:,None],proposal,d['base']);m=old.metrics(z,d['labels'],d['base'],test)
            m.update(supported=int(gate[test].sum()),seed_agreement=int(agree[test].sum()),active=int(mask[test].sum()),changed=int((z[test].argmax(-1)!=d['base'][test].argmax(-1)).sum()),threshold=float(state['threshold']))
            summaries[family]['rooms'][room]=m;torch.save(dict(logits=z,base=d['base'],support_score=scores,mask=mask,labels=d['labels'],test=test),ROUND/f'{family}_{feature}_{room}.pt')
      for item in summaries.values():
        rooms=[item['rooms'][r] for r in ('a1','c1','g1','e1')];nets=[r['corrected']-r['broken'] for r in rooms];net=sum(nets);worst=min(n/r['total'] for n,r in zip(nets,rooms));improve=sum(n>0 for n in nets)
        item.update(proxy_net=net,improving_rooms=improve,worst_room_delta=worst,eligible=net>0 and improve>=2 and worst>=-.02 and item['rooms']['actual']['correct']>=679);results.append(item)
    save(ROUND/'results.json',results)
    for r in results:print('SUPPORT_RESULT',r['family'],r['feature'],r['proxy_net'],r['improving_rooms'],r['worst_room_delta'],r['rooms']['actual']['correct'],r['eligible'],{k:(v['corrected']-v['broken'],v['active'],v['changed']) for k,v in r['rooms'].items()},flush=True)
if __name__=='__main__':main()
