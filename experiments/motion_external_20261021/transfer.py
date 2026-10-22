"""Single-antenna external pretraining plus matched source-only compute controls."""
import argparse
import gc
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from net import ROOT
from model import EvidenceEncoder
from run import WORK,FOLDS,load_data,fit,summary,save,seed_training

def train_encoder(x,y,train_indices,val_indices,folder,num_samples):
    result_path=folder/'metrics.json'
    if result_path.exists():return folder/'final.pt'
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    encoder=EvidenceEncoder(1,[64,128,256,384],256,style=True,frequency_bins=128).cuda()
    classifier=nn.Linear(256,5).cuda();seed_training()
    parameters=list(encoder.parameters())+list(classifier.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.0004,weight_decay=.01)
    yy=y[train_indices];counts=torch.bincount(yy,minlength=5)
    assert (counts>0).all(),counts
    sampler=WeightedRandomSampler((1/counts[yy]).double(),num_samples,generator=torch.Generator().manual_seed(42))
    loader=DataLoader(TensorDataset(x[train_indices],yy),batch_size=32,sampler=sampler)
    history=[]
    for epoch in range(1,6):
        encoder.train();classifier.train();losses=[]
        for xx,label in loader:
            xx,label=xx.cuda(),label.cuda();optimizer.zero_grad(set_to_none=True)
            logits=classifier(encoder(xx));loss=nn.functional.cross_entropy(logits,label)
            assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(parameters,3)
            assert torch.isfinite(norm);optimizer.step();losses.append(loss.item())
        history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
        print('PRETRAIN',folder.name,history[-1],flush=True)
    validation=None
    if val_indices:
        encoder.eval();classifier.eval();pred=[]
        with torch.no_grad():
            for start in range(0,len(val_indices),32):
                ix=val_indices[start:start+32];pred.extend(classifier(encoder(x[ix].cuda())).argmax(1).cpu().tolist())
        yy=y[val_indices];cm=torch.zeros(5,5,dtype=torch.long)
        for a,b in zip(yy.tolist(),pred):cm[a,b]+=1
        validation=dict(accuracy=float((torch.tensor(pred)==yy).float().mean()),confusion_matrix=cm.tolist(),
                        represented_classes=torch.unique(yy).tolist(),note='External date holdout; absent classes are not evaluated.')
    torch.save(dict(encoder={k:v.detach().cpu() for k,v in encoder.state_dict().items()}),folder/'final.pt')
    save(result_path,dict(history=history,validation=validation,training_samples_per_epoch=num_samples,
                          steps_per_epoch=int(np.ceil(num_samples/32)),train_class_counts=counts.tolist()))
    del encoder,classifier,optimizer;gc.collect();torch.cuda.empty_cache()
    return folder/'final.pt'

def main(args):
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available()
    external=torch.load(WORK/'external/single_rx_features_v2.pt',map_location='cpu',weights_only=True)
    rows=external['rows'];ti=[i for i,r in enumerate(rows) if r['split']=='external_train']
    vi=[i for i,r in enumerate(rows) if r['split']=='external_holdout']
    assert {rows[i]['record'] for i in ti}.isdisjoint({rows[i]['record'] for i in vi})
    n=len(ti)
    checkpoint=train_encoder(external['x'],external['y'],ti,vi,WORK/'pretrain/external',n)
    d=load_data()
    if args.full:
        fit(d,'external_transfer',42,'full',checkpoint)
        print('TRANSFER_FULL_COMPLETE',flush=True);return
    for fold in FOLDS:fit(d,'external_transfer',42,fold,checkpoint)
    save(WORK/'external_transfer_summary.json',summary('external_transfer',42))
    # This control uses only the source side of each official room split.
    x=d['t'][:,:384].reshape(-1,3,128,150).reshape(-1,1,128,150)
    y=d['y'][:,None].expand(-1,3).reshape(-1)
    for fold in FOLDS:
        source=[i*3+rx for i,r in enumerate(d['rows']) if r['room'] not in ['unknown',fold] for rx in range(3)]
        cp=train_encoder(x,y,source,[],WORK/'pretrain'/f'source_only_{fold}',n)
        fit(d,'source_pretrain_control',42,fold,cp)
    save(WORK/'source_pretrain_control_summary.json',summary('source_pretrain_control',42))
    print('TRANSFER_STAGE_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');main(p.parse_args())
