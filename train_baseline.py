"""Train the room-isolated baseline on GPU; export and check CPU ONNX."""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

import blosc2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from roomsense_model import RoomSenseModel, SubmissionModel, export_submission


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def window(csi, ati, row):
    start, stop = int(row['start']), int(row['stop'])
    x = csi[start:stop].view(np.float32).reshape(600, 1, 3, 1024, 2).copy()
    dt = ati[start:stop].copy()
    dt[-1] = 0
    return x, dt


def features(rows, model, cache, identity):
    if cache.exists():
        data = torch.load(cache, weights_only=True, map_location='cpu')
        if data['identity'] == identity:
            print(f'Using cache {cache}', flush=True)
            return data['temporal'], data['summary'], data['labels']
    temporal, summary = [], []
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(row['identifier'], []).append((i, row))
    index_order = []
    with torch.no_grad():
        for n, (ident, group) in enumerate(grouped.items()):
            csi = blosc2.load_array(group[0][1]['csi_path'])
            ati = blosc2.load_array(group[0][1]['ati_path'])
            for start in range(0, len(group), 4):
                batch = group[start:start+4]
                arrays = [window(csi, ati, row) for _, row in batch]
                x = torch.from_numpy(np.stack([a[0] for a in arrays])).cuda()
                dt = torch.from_numpy(np.stack([a[1] for a in arrays])).cuda()
                t, s = model.preprocess(x, dt)
                temporal.append(t.cpu())
                summary.append(s.cpu())
                index_order.extend(i for i, _ in batch)
            del csi, ati
            print(f'Cache {cache.stem}: {n+1}/{len(grouped)} traces', flush=True)
    order = torch.tensor(np.argsort(index_order))
    temporal = torch.cat(temporal)[order].contiguous()
    summary = torch.cat(summary)[order].contiguous()
    labels = torch.tensor([int(r['headcount']) for r in rows])
    assert torch.isfinite(temporal).all() and torch.isfinite(summary).all()
    torch.save(dict(identity=identity, temporal=temporal, summary=summary, labels=labels), cache)
    return temporal, summary, labels


def evaluate(model, loader):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for t, s, y in loader:
            logits.append(model.forward_features(t.cuda(), s.cuda()).cpu())
            labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    loss = nn.functional.cross_entropy(logits, labels).item()
    accuracy = (logits.argmax(1) == labels).float().mean().item()
    return loss, accuracy, logits.numpy()


def metrics(labels, predicted):
    cm = np.zeros((5, 5), dtype=int)
    np.add.at(cm, (labels, predicted), 1)
    return dict(accuracy=float((labels == predicted).mean()), confusion_matrix=cm.tolist(),
                recall={str(i): float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None for i in range(5)},
                support={str(i): int(cm[i].sum()) for i in range(5)})


def export_check(model, rows, folder, feature_logits):
    import onnx
    import onnxruntime as ort
    model.cpu().eval()
    first = rows[0]
    csi, ati = blosc2.load_array(first['csi_path']), blosc2.load_array(first['ati_path'])
    example = tuple(torch.from_numpy(a) for a in window(csi, ati, first))
    del csi, ati
    export_submission(model, example, folder/'model.onnx')
    onnx.checker.check_model(str(folder/'model.onnx'))
    assert (folder/'model.onnx').stat().st_size < 100_000_000
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    session = ort.InferenceSession(str(folder/'model.onnx'), sess_options=opts,
                                   providers=['CPUExecutionProvider'])
    wrapper = SubmissionModel(model).eval()
    predictions, errors, timing = [], [], []
    current, csi, ati = None, None, None
    with torch.no_grad():
        for i, row in enumerate(rows):
            if current != row['identifier']:
                csi = ati = None
                csi, ati = blosc2.load_array(row['csi_path']), blosc2.load_array(row['ati_path'])
                current = row['identifier']
            x, dt = window(csi, ati, row)
            inputs = {'frames': x, 'intervals': dt}
            if i == 0:
                session.run(None, inputs)
            start = perf_counter()
            result = session.run(['logits'], inputs)[0]
            timing.append(perf_counter()-start)
            reference = wrapper(torch.from_numpy(x), torch.from_numpy(dt)).numpy()
            assert result.shape == (5,) and result.dtype == np.float32 and np.isfinite(result).all()
            errors.append(float(abs(result-reference).max()))
            predictions.append(dict(identifier=row['identifier'], room=row['room'], start=row['start'],
                                    target=int(row['headcount']), predicted=int(result.argmax()),
                                    **{f'logit_{j}':float(v) for j,v in enumerate(result)}))
    labels = np.array([r['target'] for r in predictions])
    pred = np.array([r['predicted'] for r in predictions])
    report = metrics(labels, pred)
    report.update(onnx_bytes=(folder/'model.onnx').stat().st_size, cpu_threads=4,
                  cpu_mean_s=float(np.mean(timing)), cpu_p95_s=float(np.percentile(timing,95)),
                  max_pytorch_onnx_error=max(errors),
                  feature_gpu_onnx_prediction_agreement=float((pred==feature_logits.argmax(1)).mean()),
                  numerical_note='Float32 raw-input reductions differ across runtimes; full validation errors and predictions reported',
                  local_only=True, samples=len(rows))
    report['numerical_validation_passed'] = (report['max_pytorch_onnx_error'] < 0.01
                                            and report['feature_gpu_onnx_prediction_agreement'] == 1.0)
    assert report['cpu_mean_s'] < 0.7
    write_csv(folder/'val_predictions_onnx.csv', predictions)
    (folder/'onnx_validation.json').write_text(json.dumps(report, indent=2))
    assert report['numerical_validation_passed'], 'Numerical/export regression; inspect onnx_validation.json'
    return report


def plot(folder, history, cm):
    os.environ.setdefault('MPLCONFIGDIR', str((folder/'matplotlib_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for name in ['train_loss','val_loss']:
        axes[0].plot([h['epoch'] for h in history],[h[name] for h in history],label=name)
    for name in ['train_accuracy_online','val_accuracy']:
        axes[1].plot([h['epoch'] for h in history],[h[name] for h in history],label=name)
    axes[0].set(title='Training objective / validation CE',xlabel='Epoch',ylabel='Loss')
    axes[1].set(title='Accuracy',xlabel='Epoch',ylabel='Fraction correct',ylim=(0,1))
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'training_curves.png',dpi=150);plt.close(fig)
    fig, ax=plt.subplots(figsize=(5,4))
    image=ax.imshow(cm,cmap='Blues',vmin=0)
    for i in range(5):
        for j in range(5): ax.text(j,i,str(cm[i][j]),ha='center',va='center',color='black')
    ax.set(xlabel='Predicted headcount',ylabel='True headcount',xticks=range(5),yticks=range(5),title='Held-out room: ONNX CPU')
    fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(folder/'confusion_matrix.png',dpi=150);plt.close(fig)


def main(args):
    out=args.output
    if (out/'config.json').exists():
        raise FileExistsError(f'Run directory already used: {out}; choose another --output')
    out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA required'
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    fold=Path('analysis/splits')/args.fold
    train_rows,val_rows=read_csv(fold/'train_windows.csv'),read_csv(fold/'val_windows.csv')
    assert not {r['identifier'] for r in train_rows} & {r['identifier'] for r in val_rows}
    assert not {r['room'] for r in train_rows} & {r['room'] for r in val_rows}
    assert all(r['room']!='unknown' for r in train_rows+val_rows)
    model=RoomSenseModel().cuda()
    source_hash=hashlib.sha256(Path('roomsense_model.py').read_bytes()).hexdigest()
    config=dict(fold=args.fold,seed=args.seed,max_epochs=args.epochs,patience=12,min_epochs=20,
                batch_size=32,lr=0.001,weight_decay=0.001,label_smoothing=0.05,
                class_weighting='inverse sqrt training window frequency; mean normalized',
                selection='highest val accuracy, then lowest unweighted val CE',
                torch=torch.__version__,gpu=torch.cuda.get_device_name(0),source_sha256=source_hash,
                preprocessing='fixed features cached on GPU; no learned or dataset-fitted statistics',
                train_windows=len(train_rows),val_windows=len(val_rows),external_data=False,unlabeled_data=False)
    for part in ['train','val']:
        config[f'{part}_manifest_sha256']=hashlib.sha256((fold/f'{part}_windows.csv').read_bytes()).hexdigest()
    (out/'config.json').write_text(json.dumps(config,indent=2))
    write_csv(out/'train_windows.csv',train_rows);write_csv(out/'val_windows.csv',val_rows)
    (out/'roomsense_model_snapshot.py').write_bytes(Path('roomsense_model.py').read_bytes())
    train=features(train_rows,model,out/'train_features.pt',source_hash+config['train_manifest_sha256'])
    val=features(val_rows,model,out/'val_features.pt',source_hash+config['val_manifest_sha256'])
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(TensorDataset(*train),batch_size=32,shuffle=True,generator=generator,pin_memory=True)
    train_eval=DataLoader(TensorDataset(*train),batch_size=64,pin_memory=True)
    val_loader=DataLoader(TensorDataset(*val),batch_size=64,pin_memory=True)
    freq=torch.bincount(train[2],minlength=5).float()
    weights=freq.rsqrt();weights=weights/weights.mean()
    criterion=nn.CrossEntropyLoss(weight=weights.cuda(),label_smoothing=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=4,min_lr=1e-5)
    history=[];best=(-1.,float('-inf'));best_epoch=0
    for epoch in range(1,args.epochs+1):
        start=perf_counter();model.train();total=correct=0;loss_sum=0.
        for t,s,y in train_loader:
            t,s,y=t.cuda(),s.cuda(),y.cuda()
            optimizer.zero_grad(set_to_none=True)
            logits=model.forward_features(t,s);loss=criterion(logits,y)
            assert torch.isfinite(loss)
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.)
            assert torch.isfinite(norm)
            optimizer.step()
            total+=len(y);correct+=(logits.argmax(1)==y).sum().item();loss_sum+=loss.item()*len(y)
        val_loss,val_accuracy,_=evaluate(model,val_loader)
        scheduler.step(val_loss)
        row=dict(epoch=epoch,train_loss=loss_sum/total,train_accuracy_online=correct/total,
                 val_loss=val_loss,val_accuracy=val_accuracy,lr=optimizer.param_groups[0]['lr'],
                 elapsed_s=perf_counter()-start)
        history.append(row);write_csv(out/'history.csv',history)
        score=(val_accuracy,-val_loss)
        if score>best:
            best=score;best_epoch=epoch
            torch.save(dict(model_state=model.state_dict(),epoch=epoch,config=config,
                            val_accuracy=val_accuracy,val_loss=val_loss),out/'best.pt')
        torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                        epoch=epoch,config=config),out/'last.pt')
        print(f'Epoch {epoch:02d}: train={correct/total:.3f} val={val_accuracy:.3f} val_CE={val_loss:.4f} best={best_epoch} time={row["elapsed_s"]:.1f}s',flush=True)
        if epoch>=20 and epoch-best_epoch>=12:
            print('Early stopping',flush=True);break
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model_state']);model.cuda()
    tr_loss,tr_acc,tr_logits=evaluate(model,train_eval)
    va_loss,va_acc,va_logits=evaluate(model,val_loader)
    labels=val[2].numpy();pred=va_logits.argmax(1)
    results=metrics(labels,pred)
    majority=int(freq.argmax())
    results.update(best_epoch=best_epoch,epochs_run=len(history),train_eval_accuracy=tr_acc,
                   train_eval_ce=tr_loss,val_ce=va_loss,train_majority_class=majority,
                   train_majority_val_accuracy=float((labels==majority).mean()),
                   uniform_random_expected_accuracy=.2,parameters=sum(p.numel() for p in model.parameters()),
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   per_trace=[dict(identifier=ident,accuracy=float((pred[[r['identifier']==ident for r in val_rows]]==labels[[r['identifier']==ident for r in val_rows]]).mean())) for ident in sorted({r['identifier'] for r in val_rows})])
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    write_csv(out/'val_predictions_gpu.csv',[dict(identifier=r['identifier'],room=r['room'],start=r['start'],
                target=int(y),predicted=int(p),**{f'logit_{j}':float(x) for j,x in enumerate(logit)})
                for r,y,p,logit in zip(val_rows,labels,pred,va_logits)])
    exported=export_check(model,val_rows,out,va_logits)
    plot(out,history,exported['confusion_matrix'])
    summary=['# First supervised GPU baseline','',f'- Held-out room: {args.fold}; no external or unlabeled data.',
             f'- Best epoch: {best_epoch}/{len(history)}; training eval accuracy: {tr_acc:.2%}.',
             f'- Validation GPU accuracy: {va_acc:.2%}; CPU ONNX accuracy: {exported["accuracy"]:.2%}.',
             f'- Training-majority baseline: {results["train_majority_val_accuracy"]:.2%}; uniform random expectation: 20%.',
             f'- CPU average: {exported["cpu_mean_s"]:.4f}s/window; P95: {exported["cpu_p95_s"]:.4f}s.',
             f'- ONNX size: {exported["onnx_bytes"]:,} bytes.',
             '- Validation is used for checkpoint selection; this is not an independent test score.',
             '- One held-out room, correlated windows, and room/layout confounding limit generalization claims.',
             '- This run does not evaluate auxiliary room folds or the official hidden test.',
             '', 'See history.csv, metrics.json, onnx_validation.json and validation predictions.','']
    (out/'REPORT.md').write_text('\n'.join(summary),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)
    print(f'Completed: {out}',flush=True)


def finalize_existing(out):
    """Complete evaluation/export without retraining or changing the checkpoint."""
    torch.set_num_threads(4)
    config=json.loads((out/'config.json').read_text())
    assert hashlib.sha256(Path('roomsense_model.py').read_bytes()).hexdigest()==config['source_sha256']
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model=RoomSenseModel().eval()
    model.load_state_dict(checkpoint['model_state'])
    rows=read_csv(out/'val_windows.csv')
    predicted=read_csv(out/'val_predictions_gpu.csv')
    logits=np.array([[float(r[f'logit_{j}']) for j in range(5)] for r in predicted])
    assert [(r['identifier'],r['start']) for r in rows]==[(r['identifier'],r['start']) for r in predicted]
    result=export_check(model,rows,out,logits)
    history=[{k:float(v) for k,v in r.items()} for r in read_csv(out/'history.csv')]
    plot(out,history,result['confusion_matrix'])
    m=json.loads((out/'metrics.json').read_text())
    text=['# First supervised GPU baseline','',
          f'- Held-out fold: {config["fold"]}; no external/unlabeled data.',
          f'- Best epoch {m["best_epoch"]}; stopped after {m["epochs_run"]} epochs.',
          f'- Training eval accuracy: {m["train_eval_accuracy"]:.2%}.',
          f'- GPU validation accuracy: {m["accuracy"]:.2%}; CPU ONNX accuracy: {result["accuracy"]:.2%}.',
          f'- Training-majority baseline on validation: {m["train_majority_val_accuracy"]:.2%}; uniform random expectation: 20%.',
          f'- CPU mean {result["cpu_mean_s"]:.5f}s, P95 {result["cpu_p95_s"]:.5f}s; ONNX {result["onnx_bytes"]:,} bytes.',
          f'- Maximum PyTorch CPU / ONNX absolute logit difference: {result["max_pytorch_onnx_error"]:.6f}.',
          f'- Cached GPU / CPU ONNX prediction agreement: {result["feature_gpu_onnx_prediction_agreement"]:.2%}.',
          '- Severe train/validation gap: this baseline has not demonstrated cross-room generalization.',
          '- Validation selected the checkpoint; it is not an independent test score.',
          '- One held-out room, correlated windows, and layout/room confounding limit interpretation.',
          '- Auxiliary folds and official platform scores have not been measured.',
          '- train_loss is weighted/smoothed; val_loss is unweighted cross entropy; their scales are not identical.',
          '', 'See metrics.json, onnx_validation.json, history.csv and prediction CSV files.','']
    (out/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',default='holdout_a1')
    parser.add_argument('--output',type=Path,default=Path('artifacts/baseline_a1_seed42'))
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    finalize_existing(args.output) if args.finalize else main(args)
