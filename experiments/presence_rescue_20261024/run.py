"""Record-disjoint A/B presence rescue screen; frozen .48 is never rewritten."""
import csv
import gc
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('D:/huawei2026-experiments/presence_rescue_20261024')
PREVIOUS = Path('D:/huawei2026-experiments/protected_context_20261024')
CACHE = Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
sys.path.insert(0, str(ROOT / 'experiments/tta_policy_20261024'))
from common import build, permutation_indices, sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def extract(model, data, reverse):
    model = model.cuda().eval()
    index, signs, summary_index = [v.cuda() for v in permutation_indices((2, 1, 0))]
    features, outputs, direct, ordinal, experts = [], [], [], [], []
    with torch.inference_mode():
        for offset in range(0, len(data['y']), 8):
            t, s = data['t'][offset:offset+8].cuda(), data['s'][offset:offset+8].cuda()
            rt, rs = reverse['t'][offset:offset+8].cuda(), reverse['s'][offset:offset+8].cuda()
            pairs = [(t, s), (rt, rs), (t.index_select(1, index)*signs, s.index_select(1, summary_index)),
                     (rt.index_select(1, index)*signs, rs.index_select(1, summary_index))]
            tasks = [model.forward_tasks(a, b) for a, b in pairs]
            chunks = [torch.cat([v['direct'], v['ordinal'], v['conditional_logits'].sigmoid(),
                                v['gate'], v['experts'].softmax(-1).flatten(1), v['logits']], 1) for v in tasks]
            features.append(torch.cat([*chunks, s], 1).cpu())
            outputs.append(torch.stack([v['logits'] for v in tasks], 1).cpu())
            direct.append(tasks[0]['direct'].cpu()); ordinal.append(tasks[0]['ordinal'].cpu())
            experts.append(tasks[0]['experts'].softmax(-1).cpu())
    return dict(features=torch.cat(features), logits=torch.cat(outputs), direct=torch.cat(direct),
                ordinal=torch.cat(ordinal), experts=torch.cat(experts))


def signal_input(data):
    # Uses only deterministic raw-signal-derived preprocessing, no trained encoder.
    result = []
    for offset in range(0, len(data['y']), 32):
        t = data['t'][offset:offset+32]
        a = t[:, :768].reshape(-1, 6, 128, 150)
        static = torch.cat([a[:, :3].mean(-1), a[:, :3].std(-1, unbiased=False)], 1)
        static = F.avg_pool1d(static, 4).flatten(1)
        dynamic = F.avg_pool2d(a[:, 3:], (4, 3))
        phase = F.avg_pool2d(t[:, 768:960].reshape(-1, 6, 32, 150), (1, 3))
        result.append(torch.cat([static, torch.cat([dynamic, phase], 1).flatten(1), data['s'][offset:offset+32]], 1))
    return torch.cat(result)


class Readout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Dropout(.2), nn.Linear(32, 1))
    def forward(self, x):
        return self.body(x).flatten()


class SignalBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(192+17, 64), nn.SiLU(), nn.Dropout(.25))
        self.dynamic = nn.Sequential(nn.Conv2d(9, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.SiLU(),
                                     nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Dropout(.25), nn.Linear(32, 1))
    def forward(self, x):
        static = self.static(torch.cat([x[:, :192], x[:, -17:]], 1))
        dynamic = x[:, 192:-17].reshape(-1, 9, 32, 50)
        if self.training:
            flip = torch.rand(len(x), 1, 1, 1, device=x.device) < .5
            dynamic = torch.where(flip, dynamic.flip(-1), dynamic)
        z = self.dynamic(dynamic); mean = z.mean((2, 3))
        std = (z.sub(mean[:, :, None, None]).square().mean((2, 3)) + 1e-5).sqrt()
        return self.head(torch.cat([static, mean, std], 1)).flatten()


def partition(rows, room):
    manifest = read(PREVIOUS / 'results' / f's42_{room}' / 'manifest.json')
    groups = [{r['identifier'] for r in manifest[name]} for name in ('fit', 'calibration', 'test')]
    assert all(groups[a].isdisjoint(groups[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    return [[i for i, r in enumerate(rows) if r['identifier'] in ids] for ids in groups]


def proxy_prediction(outputs, room):
    cal = read(PREVIOUS / 'results' / f's42_{room}' / 'metrics.json')['calibration']['baseline']
    temperature, bias = cal['temperature'], torch.tensor(cal['bias'])
    def corrected(logp):
        return logp / temperature + bias
    original = corrected(outputs[:, 0]).argmax(-1)
    q = outputs.exp().reshape(-1, 2, 2, 5).mean(2)
    identity = corrected(q[:, 0].clamp_min(1e-12).log())
    average = corrected(q.mean(1).clamp_min(1e-12).log())
    legacy = torch.where(((identity.argmax(-1)==0)|(identity.argmax(-1)==2))[:, None], identity, average).argmax(-1)
    return torch.where(original == 1, original, legacy)


def metrics(prob, threshold, y, base, ix, rows):
    ix = torch.tensor(ix)
    y, base, prob = y[ix], base[ix], prob[ix]
    pred = torch.where((base == 0) & (prob >= threshold), torch.ones_like(base), base)
    binary = y <= 1
    cm = torch.bincount(y*5+pred, minlength=25).reshape(5,5)
    records = []
    for identifier in sorted({rows[i]['identifier'] for i in ix.tolist()}):
        local = torch.tensor([j for j, i in enumerate(ix.tolist()) if rows[i]['identifier']==identifier])
        records.append(dict(identifier=identifier, truth=int(y[local[0]]), windows=len(local),
                            corrected=int(((pred[local]==y[local])&(base[local]!=y[local])).sum()),
                            broken=int(((pred[local]!=y[local])&(base[local]==y[local])).sum()),
                            auxiliary_present=int((prob[local]>=threshold).sum())))
    return dict(windows=len(ix), baseline_correct=int((base==y).sum()), correct=int((pred==y).sum()),
                corrected=int(((base!=y)&(pred==y)).sum()), broken=int(((base==y)&(pred!=y)).sum()),
                empty_count=int((y==0).sum()), one_count=int((y==1).sum()),
                rescued_one=int(((y==1)&(base==0)&(pred==1)).sum()),
                added_empty_false_positive=int(((y==0)&(base==0)&(pred==1)).sum()),
                standalone_binary_correct=int(((prob[binary]>=threshold)==y[binary].bool()).sum()),
                binary_count=int(binary.sum()), confusion_matrix=cm.tolist(), records=records)


def train(x, y, rows, fit_ix, cal_ix, kind, initialization, folder):
    train_ix = torch.tensor([i for i in fit_ix if y[i]<=1])
    calibration = torch.tensor([i for i in cal_ix if y[i]<=1])
    assert set(y[train_ix].tolist())==set(y[calibration].tolist())=={0,1}
    seed(initialization)
    mean = x[train_ix].mean(0); scale=x[train_ix].std(0,unbiased=False).clamp_min(.1)
    if kind=='B_signal':
        # The dynamic/phase images are already per-input normalized; do not fit
        # pixel-wise room patterns as a second normalization.
        mean[192:-17]=0; scale[192:-17]=1
    normalized=((x-mean)/scale).clamp(-8,8).cuda()
    labels=y.float().cuda()
    model=(SignalBranch() if kind=='B_signal' else Readout(x.shape[1])).cuda()
    groups={}
    for i in train_ix.tolist():
        key=(int(y[i]),rows[i]['room']); groups[key]=groups.get(key,0)+1
    domains={label:len({room for value,room in groups if value==label}) for label in (0,1)}
    weights=torch.tensor([1/(groups[(int(y[i]),rows[i]['room'])]*domains[int(y[i])]) for i in train_ix]).cuda()
    weights=weights/weights.sum()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    train_gpu=train_ix.cuda(); history=[]
    for epoch in range(40):
        model.train(); losses=[]
        order=torch.multinomial(weights,len(train_ix),replacement=True)
        for selected in order.split(64):
            batch=train_gpu[selected]
            loss=F.binary_cross_entropy_with_logits(model(normalized[batch]),labels[batch])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3); optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    model.eval()
    with torch.inference_mode():
        probabilities=torch.cat([model(v).sigmoid().cpu() for v in normalized.split(64)])
    negatives=probabilities[calibration[y[calibration]==0]]
    # Fixed safety rule, no use of outer room labels or threshold grid search.
    threshold=max(.5,float(negatives.max())+1e-6)
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state={k:v.cpu() for k,v in model.state_dict().items()},mean=mean,scale=scale,
                    kind=kind,seed=initialization,threshold=threshold,
                    fit_identifiers=sorted({rows[i]['identifier'] for i in train_ix.tolist()}),
                    calibration_identifiers=sorted({rows[i]['identifier'] for i in calibration.tolist()})),folder/'final.pt')
    save(folder/'training.json',dict(loss=history,epochs=40,parameter_count=sum(p.numel() for p in model.parameters()),
        threshold=threshold,calibration_empty_windows=len(negatives),
        calibration_one_detected=int((probabilities[calibration[y[calibration]==1]]>=threshold).sum())))
    del model,normalized,optimizer; gc.collect(); torch.cuda.empty_cache()
    return probabilities,threshold


def main():
    torch.set_num_threads(4); torch.backends.cudnn.deterministic=True
    WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(seeds=[42,2026],rooms=['a1','c1','g1','e1'],epochs=40,
        methods=['A_actual_diagnostic','A_proxy','B_signal'],
        split='reuse pre-existing record-disjoint fit/calibration/outer-room manifests; unknown excluded',
        threshold='max(0.5, maximum calibrated empty probability +1e-6); never use outer labels',
        gate='only baseline output0 may change to1',
        limitations=['actual .48 encoder saw all source labels; A_actual is diagnostic only',
                     'A_proxy uses independent historical outer-room-excluded seed42 backbones; two seeds apply to new heads only',
                     'Repeatedly used development rooms, tiny record count and one empty calibration record per fold',
                     'B has independent raw-derived features; gated actual .48 performance still not independent whole pipeline'],
        frozen_model_sha256=sha(ROOT/'models/original_one_guard_20261024/01_original_one_guard.onnx')))
    data=torch.load(CACHE,map_location='cpu',weights_only=True); data.pop('absolute',None)
    reverse=torch.load(PREVIOUS/'reverse_features.pt',map_location='cpu',weights_only=True)
    assert reverse['cache_sha256']==sha(CACHE)
    assert reverse['rows']==data['rows']
    rows,y=data['rows'],data['y']
    joined=read(Path('D:/huawei2026-experiments/remaining048_20261024/joined_windows.json'))
    lookup={(r['identifier'],int(r['start'])):r for r in joined}
    actual=torch.tensor([lookup[r['identifier'],int(r['start'])]['prediction'] for r in rows])
    assert int((actual==y).sum())==654
    actual_cache=WORK/'actual_evidence.pt'
    if actual_cache.exists(): evidence=torch.load(actual_cache,weights_only=True)
    else:
        model,bias=build(ROOT/'models/best_037/final.pt',full=True)
        evidence=extract(model,data,reverse); evidence['bias']=bias
        torch.save(evidence,actual_cache); del model; gc.collect(); torch.cuda.empty_cache()
    # Check reconstructed original view against previously executed actual ONNX.
    with (Path('D:/huawei2026-experiments/actual047_audit_20261024/all_predictions.csv')).open(encoding='utf-8-sig',newline='') as f:
        audit={(r['identifier'],int(r['start'])):r for r in csv.DictReader(f)}
    expected=torch.tensor([[float(audit[r['identifier'],int(r['start'])][f'base044_logit_{k}']) for k in range(5)] for r in rows])
    rebuilt=evidence['logits'][:,0]+evidence['bias']
    prob_error=float((rebuilt.softmax(-1)-expected.softmax(-1)).abs().max())
    disagreements=int((rebuilt.argmax(-1)!=expected.argmax(-1)).sum())
    # Cached preprocessed features are not bit-identical to raw-input ONNX.
    # CPU spot checks on the four largest differences reproduce the discrepancy
    # (GPU-vs-CPU <=5.2e-5); retain only a diagnostic, not an export equivalence.
    assert prob_error<.005 and disagreements==0,(prob_error,disagreements)
    remaining=(actual!=y); missed=(actual==0)&(y==1)
    decomposition=dict(reconstructed_probability_max_error=prob_error,argmax_disagreements=disagreements,
        note='PyTorch frozen weights on cached features: all original-view classes agree with actual ONNX, but probability max discrepancy up to0.005; diagnostic only, not export equivalence',
        remaining_errors=int(remaining.sum()),one_missed_as_empty=int(missed.sum()))
    for name, values in [('direct',evidence['direct']),('ordinal',evidence['ordinal'])]:
        decomposition[name]=dict(correct_on_remaining=int(((values.argmax(-1)==y)&remaining).sum()),
            supports_one_on_missed=int(((values.argmax(-1)==1)&missed).sum()))
    decomposition['presence_probability_on_missed']=evidence['ordinal'][missed,0].neg().add(1).quantile(torch.tensor([0.,.25,.5,.75,1.])).tolist()
    decomposition['static_expert_supports_one_on_missed']=int(((evidence['experts'][:,0].argmax(-1)==1)&missed).sum())
    save(WORK/'head_decomposition.json',decomposition)
    print('HEAD_DECOMPOSITION',decomposition,flush=True)
    signal=signal_input(data)
    results=[]
    for room in ('a1','c1','g1','e1'):
        fit_ix,cal_ix,test_ix=partition(rows,room)
        proxy_path=WORK/f'proxy_evidence_{room}.pt'
        if proxy_path.exists(): proxy=torch.load(proxy_path,weights_only=True)
        else:
            checkpoint=PREVIOUS/'fits'/f'baseline_s42_{room}'/'final.pt'
            cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert cp['fit_identifiers']==sorted({rows[i]['identifier'] for i in fit_ix})
            model,_=build(checkpoint)
            proxy=extract(model,data,reverse); proxy['checkpoint_sha256']=sha(checkpoint)
            torch.save(proxy,proxy_path); del model,cp; gc.collect(); torch.cuda.empty_cache()
        proxy_base=proxy_prediction(proxy['logits'],room)
        save(WORK/'splits'/f'{room}.json',dict(fit=[rows[i] for i in fit_ix],calibration=[rows[i] for i in cal_ix],test=[rows[i] for i in test_ix]))
        for initialization in (42,2026):
            for kind,x in [('A_actual_diagnostic',evidence['features']),('A_proxy',proxy['features']),('B_signal',signal)]:
                folder=WORK/'fits'/f'{kind}_{room}_s{initialization}'
                probability,threshold=train(x,y,rows,fit_ix,cal_ix,kind,initialization,folder)
                result=dict(room=room,seed=initialization,kind=kind,threshold=threshold,
                            actual048=metrics(probability,threshold,y,actual,test_ix,rows),
                            independent_backbone_proxy=metrics(probability,threshold,y,proxy_base,test_ix,rows))
                np.savez_compressed(folder/'predictions.npz',probability=probability.numpy(),labels=y.numpy(),
                                    actual048=actual.numpy(),proxy_base=proxy_base.numpy())
                save(folder/'result.json',result); results.append(result)
                print('COMPLETE',room,initialization,kind,'threshold',threshold,
                      'actual corrected/broken',result['actual048']['corrected'],result['actual048']['broken'],
                      'proxy corrected/broken',result['independent_backbone_proxy']['corrected'],result['independent_backbone_proxy']['broken'],flush=True)
        del proxy; gc.collect()
        save(WORK/'results.json',results)
    assert sha(ROOT/'models/original_one_guard_20261024/01_original_one_guard.onnx')==read(WORK/'plan.json')['frozen_model_sha256']
    print('SCREEN_FINISHED',flush=True)


if __name__=='__main__': main()
