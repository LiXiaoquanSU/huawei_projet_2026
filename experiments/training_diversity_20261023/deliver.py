"""Export the prespecified-seed training-diversity partner and fixed blend."""
import csv
import gc
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from train import ROOT, WORK, OLD, Model, ref


class Submission(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model

    def forward(self, frames, intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_features(t.float(),s.float())[0]


def main():
    torch.set_num_threads(4)
    selection=ref.read(WORK/'selection.json')
    assert ref.read(WORK/'complete.json')['fits']==15
    audits=[]
    for metrics_path in sorted((WORK/'fits').glob('*/metrics.json')):
        metric=ref.read(metrics_path);config=metric['config'];fold=config['fold']
        assert (config['epochs'],config['batch_size'],config['lr'])==(5,32,.0004)
        with (metrics_path.parent/'train_manifest.csv').open(encoding='utf-8-sig') as stream:
            train_rows=list(csv.DictReader(stream))
        if fold!='full':
            with (metrics_path.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as stream:
                valid_rows=list(csv.DictReader(stream))
            assert {r['identifier'] for r in train_rows}.isdisjoint({r['identifier'] for r in valid_rows})
            assert all(r['room'] not in ('unknown',fold) for r in train_rows)
        else:
            assert len({r['identifier'] for r in train_rows})==48
        parity=None
        if config['architecture']=='physical' and config['seed']==2026 and fold in ('a1','c1','g1'):
            prior=ref.read(OLD/'fits'/f'context_events_s2026_{fold}/metrics.json')
            parity=metric['validation']['confusion_matrix']==prior['validation']['confusion_matrix']
            assert parity,'Original-training control failed parity'
        audits.append(dict(fit=metrics_path.parent.name,record_disjoint=True,original_control_parity=parity))
    assert len(audits)==15
    ref.save(WORK/'audit.json',dict(status='pass',fits=audits))
    recipe=selection['recipe'];seed=selection['seed']
    folder=WORK/'fits'/f'{recipe}_s{seed}_full'
    cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    assert ref.read(folder/'metrics.json')['actual_records']==48
    model=Model(recipe).eval();model.load_state_dict(cp['model_state'])
    wrapper=Submission(model).eval()
    export=WORK/'export';export.mkdir(exist_ok=True)
    # Load the graph composition implementation used for the platform .40 model.
    spec=importlib.util.spec_from_file_location('dual_export',ROOT/'experiments/dual039_20261022/build.py')
    dual=importlib.util.module_from_spec(spec);spec.loader.exec_module(dual)
    sources=[*list((ROOT/'src').glob('*.py')),
             ROOT/'experiments/global_local_20261021/network.py',
             ROOT/'experiments/structure_events_20261021/network.py',
             ROOT/'experiments/structure_events_20261021/train.py',
             ROOT/'experiments/dual039_20261022/build.py',
             ROOT/'tools/check_candidate.py',
             *list(Path(__file__).parent.glob('*.py'))]
    manifest=[]
    for source in sources:
        relative=source.relative_to(ROOT);snapshot=WORK/'source_snapshot'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,snapshot)
        manifest.append(dict(path=relative.as_posix(),sha256=dual.sha(source)))
    ref.save(WORK/'source_manifest.json',manifest)
    original_refs=ref.read(OLD/'exports/cross_events_s42/references.json')
    references=[];partner_refs=[]
    for item in original_refs:
        if item['input']=='zero':
            x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(item['input']) as data:
                x,dt=[torch.from_numpy(data[k].copy()) for k in ('frames','intervals')]
        with torch.no_grad():output=wrapper(x,dt).numpy()
        partner_refs.append(dict(input=item['input'],logits=output.tolist()))
        p=(dual.probability(output)+dual.probability(item['logits']))*.5
        references.append(dict(input=item['input'],logits=np.log(np.maximum(p,1e-12)).tolist()))
        if len(references)==1:example=x,dt
    torch.onnx.export(wrapper,example,str(export/'partner.onnx'),input_names=['frames','intervals'],
                      output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    targetdir=ROOT/'models/training_diversity_20261023';targetdir.mkdir(exist_ok=True)
    partner=targetdir/f'02_context_{recipe}_seed314.onnx'
    assert not partner.exists()
    shutil.copy2(export/'partner.onnx',partner)
    # Only storage is rounded, computation types remain unchanged.
    dual.SPECS=[dual.SPECS[0],('context_'+recipe,314,'../training_diversity_20261023/'+partner.name,dual.sha(partner))]
    compression=dual.assemble(export)
    ref.save(export/'references.json',references)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(export)],check=True)
    # Candidate rounding sensitivity on cached inputs, not an accuracy estimate.
    d=ref.load();model=model.cuda()
    before=[];after=[]
    with torch.no_grad():
        for start in range(0,len(d['y']),32):
            before.append(model.forward_features(d['t'][start:start+32].cuda(),d['s'][start:start+32].cuda()).softmax(-1).cpu())
        for value in model.state_dict().values():
            if value.dtype==torch.float32 and value.numel()>=1024:
                value.copy_(value.half().float())
        for start in range(0,len(d['y']),32):
            after.append(model.forward_features(d['t'][start:start+32].cuda(),d['s'][start:start+32].cuda()).softmax(-1).cpu())
    before,after=torch.cat(before),torch.cat(after)
    error=float((before-after).abs().max());changes=int((before.argmax(1)!=after.argmax(1)).sum())
    assert error<.002
    rounding=dict(windows=len(before),probability_max_error=error,argmax_changes=changes,
                  note='partner weight-rounding numerical audit, not held-out accuracy')
    resource=ref.read(export/'resource_check.json')
    protected=ROOT/'models/best_040/model.onnx'
    assert dual.sha(protected)=='c129649a841d103975ea481254bd303b87810179e1789166f72dbb42fbc2b250'
    target=targetdir/f'01_cross039_context_{recipe}_seed314_equalprob.onnx'
    assert not target.exists();shutil.copy2(export/'model.onnx',target)
    alias=ROOT/'submissions/training_diversity_20261023/model.onnx';alias.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(target,alias)
    meta=dict(path=target.relative_to(ROOT).as_posix(),alias=alias.relative_to(ROOT).as_posix(),
              sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,status='research_candidate_platform_unverified',
              architecture='original cross42 + freshly trained context '+recipe,seed=314,weights=[.5,.5],
              selection=selection,resource=resource,rounding_audit=rounding,compression=compression,
              checkpoint=str(folder/'final.pt'),full_training=ref.read(folder/'metrics.json'),
              stress_tests={x:ref.read(WORK/'fits'/f'{x}_s314_e1/metrics.json')['validation'] for x in ('physical','direct')},
              report='docs/TRAINING_DIVERSITY_20261023.md')
    for p in (target.with_suffix('.json'),alias.with_suffix('.json'),WORK/'candidate.json'):
        ref.save(p,meta)
    # Preserve the original best; new score remains unknown until reported.
    registry=ref.read(ROOT/'models/registry.json');registry['training_diversity_20261023']={k:meta[k] for k in ('path','alias','sha256','bytes','platform_score','status','report')}
    registry['recommendations']['next_submission']='training_diversity_20261023'
    registry['recommendations']['next_submission_note']='Training-diversity fixed blend; platform untested. Original .40 remains best.'
    ref.save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows.append(dict(model='cross039 + context '+recipe+' seed314 equal probability',platform_score='',path=meta['path'],
                     sha256=meta['sha256'],source='paired training-loss experiment; platform untested',status=meta['status'],submission_id=''))
    with p.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=[]
    for item in selection['summary']:
        lines.append(f"|{item['recipe']}|{item['room_mean']:.2%}|{item['macro_recall']:.2%}|{item['worst_room']:.2%}|")
    report=f'''# 训练差异与互补模型实验

固定 context_events 成功结构，对比 physical 原复合损失与 direct 直接分类。direct = 五分类 CE + 0.15 专家 CE，去掉序数、JS 与信号辅助损失，同时推理只用直接分类概率，避免未训练的序数分支污染输出。其余数据、采样、5轮训练、优化器和每轮更新数保持原配方。

每臂 seed314、2026 × a1/c1/g1，12 次拟合；选定配方后固定 seed314 部署，两个配方另做 e1 压力测试，最后选中配方全48记录训练，共15次。训练/验证按源记录和房间隔离；unknown 不进开发。没有使用外部数据或密集滑窗。没有挑最高验证epoch。

比较目标是新 context 与原 cross42 固定等权融合，与原 cross42/context2026 组成的0.40模型在历史折上比较。选择依据为两种子平均 min(房间均值,宏召回)，种子314预先固定。历史折重复使用，不能当新独立测试。

|配方|融合房间均值|融合宏召回|最差房间|
|---|---:|---:|---:|
{chr(10).join(lines)}

原0.40固定组合在相同历史折均值29.49%。新候选选中 `{recipe}`；详细逐房间纠错/破坏数量、混淆矩阵、e1压力测试见本地 JSON 和 D 盘 selection.json。两种子对比中原 cross42固定，所以并不是两个独立验证集。e1压力测试单模型准确率：physical 21.67%、direct 26.67%；该结果在选配方后产生，没有反向用于挑选，也不代表完整融合模型的e1表现。

新候选：`{meta['path']}`，{meta['bytes']/1e6:.2f} MB。SHA256 `{meta['sha256']}`。新模型平台未知，不赋予0.40成绩，不保证提高0.05。原最佳完全保留。

采用已测0.40相同的半精度权重存储/还原float32计算方案。三输入CPU资源和数值检查通过；另960窗新搭档舍入数值检查最大概率差{error:.7f}，类别变化{changes}个；此处不是准确率评估。

复现顺序：`python experiments/training_diversity_20261023/train.py`，随后 `python experiments/training_diversity_20261023/deliver.py`。需要既有D盘原特征缓存、原始最佳模型、历史验证预测和smoke/gap检查输入。所有拟合记录在 `D:/huawei2026-experiments/training_diversity_20261023`，权重不重复上传GitHub。
'''
    (ROOT/'docs/TRAINING_DIVERSITY_20261023.md').write_text(report,encoding='utf-8')
    print('DELIVERED',json.dumps({k:meta[k] for k in ('path','sha256','bytes')}),flush=True)
    del model,d,cp;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    main()
