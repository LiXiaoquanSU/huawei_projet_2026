"""Preserve five user-scored artifacts and audit their executable policy."""
import csv,gc,json,shutil,hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper,helper,TensorProto
from inspect_teammate047 import structural,session,save,sha
import onnxruntime as ort
import blosc2
RAW_PROBES={}

ROOT=Path(__file__).resolve().parents[1]
INBOX=Path('C:/Users/lxq_1/xwechat_files/wxid_6ct208ti8tam12_cc30/msg/file/2026-10')
DEST=ROOT/'models/teammate054_20261025'
SPECS=[('gprime','gprime_full_group_fallback_web054.onnx',.54),
 ('gcal','gcal_calibrated_fallback_web053(1).onnx',.53),
 ('c1prime','c1prime_room_offset_web046(1).onnx',.46),
 ('ntprime','ntprime_tilt_equalisation_web045(1).onnx',.45),
 ('gnp','gnp_nongeometric_fallback_web053(1).onnx',.53)]

def logmean(z):return np.log(np.mean(np.exp(np.stack(z).astype(np.float64)),axis=0))

def probes(path,kind):
    if not RAW_PROBES:
        for label,folder,pattern in [('labeled_three','labeled','CSI_a_a1_hash033659_3.b2'),('official_u','unlabeled','CSI_*.b2')]:
            f=next(Path('D:/huawei2026-data',folder).glob(pattern));csi=blosc2.load_array(str(f));ati=blosc2.load_array(str(f.with_name(f.name.replace('CSI_','ATI_',1)))).reshape(-1)
            feed=dict(frames=np.ascontiguousarray(csi[:600]).view(np.float32).reshape(600,1,3,1024,2).copy(),intervals=ati[:600].astype(np.float32,copy=True));feed['intervals'][-1]=0
            RAW_PROBES[label]=feed;del csi,ati;gc.collect()
    m=onnx.load(path)
    if kind=='c1prime':names=['__c1_base_logits','__c1_delta','__c1_room','__c1_gate']
    else:names=['__rxg_012___tta_v0_logits']+['__rxg_'+p+'_logits' for p in ('012','021','102','120','201','210')]
    for name in names:m.graph.output.append(helper.make_tensor_value_info(name,TensorProto.INT64 if name=='__c1_room' else TensorProto.BOOL if name=='__c1_gate' else TensorProto.FLOAT,None))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    s=ort.InferenceSession(m.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);del m
    results=[]
    for file in (ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero',*RAW_PROBES):
        if file in RAW_PROBES:feed=RAW_PROBES[file]
        elif file=='zero':feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
        else:
            with np.load(file) as d:feed={k:d[k].copy() for k in ('frames','intervals')}
        out,*v=s.run(None,feed)
        if kind=='c1prime':expected=v[0]+v[1];branch='room_offset' if bool(v[3]) else 'unchanged'
        else:
            original,orders=v[0],v[1:];identity=orders[0]
            if kind=='gnp':fallback=logmean(orders[1:5])
            elif kind=='gcal':
                graph=onnx.load(path);bias={t.name:numpy_helper.to_array(t) for t in graph.graph.initializer if t.name.startswith('__gcal_bias_')};del graph
                fallback=logmean([a+bias['__gcal_bias_'+p] for a,p in zip(orders,('012','021','102','120','201','210'))])
            else:fallback=logmean(orders)
            branch='original_one' if original.argmax()==1 else 'identity_time_02' if identity.argmax() in (0,2) else 'fallback'
            expected=(original if branch=='original_one' else identity if branch=='identity_time_02' else fallback)+np.array([0,.35,0,0,0])
        err=float(np.abs(out-expected).max());assert err<2e-5,(kind,err)
        assert out.shape==(5,) and out.dtype==np.float32 and np.isfinite(out).all()
        results.append(dict(input=str(file),predicted_class=int(out.argmax()),branch=branch,formula_logit_max_error=err,logits=out.tolist()))
    del s;gc.collect();return results

def main():
    DEST.mkdir(parents=True,exist_ok=True);base=onnx.load(ROOT/'models/best_044/model.onnx');initializers={t.name:t for t in base.graph.initializer};nodes=list(base.graph.node);records=[]
    for kind,filename,score in SPECS:
        source=INBOX/filename;dest=DEST/filename.replace('(1)','');digest=sha(source)
        if dest.exists():assert sha(dest)==digest
        else:shutil.copy2(source,dest)
        assert sha(dest)==digest
        proof_path=ROOT/f'checks/teammate054_{kind}_structure.json'
        if dest.with_suffix('.json').exists() and proof_path.exists() and json.loads(dest.with_suffix('.json').read_text(encoding='utf-8')).get('sha256')==digest:
            proof=json.loads(proof_path.read_text(encoding='utf-8'));probe=json.loads(dest.with_suffix('.json').read_text(encoding='utf-8'))['audit']['formula_probes']
            if len(probe)<5:probe=probes(dest,kind)
        else:proof=structural(dest,initializers,nodes);probe=probes(dest,kind)
        assert proof['exact_network_body_copies']==(4 if kind=='c1prime' else 12)
        graph=onnx.load(dest);assert not any(t.data_location==TensorProto.EXTERNAL for t in graph.graph.initializer);del graph
        meta=dict(id='teammate_'+kind+'_20261025',name=dest.stem,path=dest.relative_to(ROOT).as_posix(),original_path=str(source),sha256=digest,bytes=dest.stat().st_size,
          platform_score=score,status='current_best' if score==.54 else 'scored_comparison',owner='teammate',source='user-reported platform score encoded in filename; 2026-10-25',submission_id=None,report='docs/TEAMMATE054_20261025.md',
          audit=dict(old044_initializers_identical=proof['old044_initializers_identical'],exact_network_body_copies=proof['exact_network_body_copies'],nodes=proof['nodes'],embedded_weights=True,formula_probes=probe),
          note='Embedded metadata may say not evaluated or stale older score; preserved untouched. Latest score comes from explicit human feedback. Probe cases do not establish accuracy; no fresh training performed.')
        save(dest.with_suffix('.json'),meta);save(ROOT/f'checks/teammate054_{kind}_structure.json',proof);records.append(meta)
        print('RECORDED',kind,score,digest,flush=True)
    registry=json.loads((ROOT/'models/registry.json').read_text(encoding='utf-8'));previous=registry['current_best']
    if previous.get('id')!=records[0]['id']:registry['previous_best_before054']=previous
    registry['current_best']=records[0];registry['tied_current_best']=[records[0]];registry['updated_date']='2026-10-25'
    ids={x['id'] for x in records};registry['retained_models']=[x for x in registry.get('retained_models',[]) if x.get('id') not in ids]+records
    # Existing schema points score_history to the CSV; preserve that reference.
    registry['teammate054_feedback']=records
    registry['target_status']=dict(target=.54,status='achieved_by_user_reported_platform_score',model=records[0]['path'],date='2026-10-25')
    registry['research_status']='teammate054_scored_and_structurally_audited'
    registry['latest_research']=dict(report='docs/TEAMMATE054_20261025.md',status='five_scored_models_audited',best_scored=.54)
    registry['candidate_note']='Team best054 reached. Previous encoder-finetuning candidate remains unscored; it is a comparison experiment, not the best-known model.'
    if registry.get('next_submission'):registry['next_submission']['recommendation_context']='Unscored comparison candidate created before teammate054 feedback; not claimed to exceed054.'
    save(ROOT/'models/registry.json',registry)
    cp=ROOT/'models/platform_scores.csv'
    with cp.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    digests={x['sha256'] for x in records};rows=[x for x in rows if x['sha256'] not in digests]
    for m in records:
        row={k:'' for k in fields};row.update(model=m['name'],platform_score=m['platform_score'],path=m['path'],sha256=m['sha256'],source=m['source'],status=m['status'],availability='retained',original_path=m['original_path']);rows.append(row)
    with cp.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    save(ROOT/'checks/teammate054_20261025.json',dict(models=records,analysis='No new training. Same178 original044 tensors and exact network bodies. Protected routing and fallback-view set change.',score_provenance='explicit user feedback, not inferred from embedded metadata'))
    table='\n'.join(f"|{m['name']}|{m['platform_score']:.2f}|{m['bytes']/1e6:.2f}MB|" for m in records)
    report=f'''# 队友五份模型登记与机制分析（2026-10-25）

## 已确认的成绩

成绩依据用户明确说明“文件名末尾是成绩”，未独立访问平台。文件均已从微信目录复制到项目，逐字节SHA256核对；原ONNX元数据仍有not evaluated、0.47等历史内容，原件不修改，以本次人类反馈为最新成绩。

|模型|平台分数|大小|
|---|---:|---:|
{table}

当前团队最佳`{records[0]['path']}`，SHA256 `{records[0]['sha256']}`。原目标0.54已由队友模型达到。原自研最高0.49仍保留；前一轮微调候选未评分，不因为这次反馈删除或伪登记其成绩。

## 从文件中直接证明的部分

五份都保留原0.44模型178个初始化张量，序列化字节完全一致。Gprime、Gcal、NTprime、GNP含12份与原0.44完全一致的网络主体；C1prime含4份。它们共享同一组网络权重，不能理解为12个独立训练模型。没有证据表明这些文件重训了主干；新增常量/偏置表不等同神经网络权重更新。五个输入探针（普通、间隔样例、零输入、标注3人原始窗口、官方U原始窗口）已实际运行，各自输出规则的手工重建误差小于2e-5，输出有限且接口正常。未做五份模型的全量准确率重测或平台运行时间复核。

## 0.54的精确规则

记v0为原始通道顺序、正向时间的原0.44输出；v45为原始通道顺序下正向与反向时间融合；F12为六种Rx排列各自正反时间融合后的总体融合。

```text
若 argmax(v0) == 1：使用 v0
否则若 argmax(v45) 为0或2：使用 v45
否则：使用 F12
最后所有分支都加 [0, 0.35, 0, 0, 0]
```

融合的实际计算是`log(mean(exp(scores)))`，不是直接算logits算术平均；scores已包含原Udirect类别偏置，因此也不应称为每个视角分别softmax归一化后再平均。

与原0.49 `01_final_bias_one035.onnx`相比，保护条件和最终偏置相同，**需要fallback时从012/210两种Rx排列×两个时间方向的4视角，扩成6种Rx排列×两个时间方向的12视角**。保护仍在原偏置分数上判断，额外+0.35在选择分支之后。0.54图中还有历史current47/outer保护节点，但该旧输出未被最终logits消费；不能照读旧元数据误以为存在额外外层保护。

因此，这一组0.49→0.54最直接支持的是“保留原有类别条件路由后，扩大fallback视角集合有效”，而不是模型变大或重新训练更有效。过去不加保护、把所有样本统一12视角融合的版本只有0.41，二者策略不同，不能据此否定12视角。

## 另外四份的变化

- **Gcal 0.53**：在Gprime基础上，fallback的每种Rx排列额外加一个5类偏置向量，再融合。保护分支依然使用原v0/v45；未改变它们。模型元数据称这些偏置来自U的无标签平衡校准，但具体拟合流程和数据无法仅凭ONNX验证。可确认新增的是六组不同偏置，不是一个全局偏置。比Gprime低0.01，未显示额外优势。
- **GNP 0.53**：保护相同，fallback只融合021、102、120、201四种Rx排列及各自正反时间，排除012、210（代码称non-geometric，名称不是已确认的物理几何结论）。与全部六种相比低0.01；说明另外四种排列有用，全六种在本次成绩最好。0.01的差别不宜当作稳定显著结论。
- **C1prime 0.46**：在原0.49四视角图后额外加入房间识别与输出偏置。它从平均幅度的对数频谱中提取384维背景指纹，标准化、线性变换、匹配11组中心；仅最近中心属于U组且距离不超过约644.86时加该组5类偏置。不是新的动作/人数编码器，也不能保证识别出的中心就是隐藏物理房间。相对原0.49低0.03，说明这个房间条件校正没有带来预期迁移。
- **NTprime 0.45**：在Gprime输入前增加按三个Rx估计对数幅度频谱斜率、向指定目标斜率调整的预处理。权重不变，但网络看到的原始输入已被改写；比Gprime低0.09。该结果反对把频谱斜率简单当成可去掉的干扰项，不能证明所有频谱归一化都无效。

## 成功原因：证据与推断分开

**可确认**：同样的主干权重、原Udirect偏置与最终1人+0.35，主要区别就是条件融合的视角集合。高分不是由新标注数据、外部数据或扩大训练网络容量直接产生。

**合理解释，尚非逐样本因果证明**：旧网络对Rx排列较敏感；不同排列产生互补判断，扩大融合可以减小特定排列的偶然偏差。时间反转同样提供另一种观察顺序。对v0判1、v45判0/2的样本保留专门路径，避免无差别平均稀释它们。天线排列并非任意改变都物理等价，尤其相位符号和天线对关系，所以要称“经平台验证有用的视角变换”，不能声称理论上严格不变。

没有隐藏标签和逐条平台预测，尚不知道提升主要来自哪个人数、哪个房间，也不能断言修复了本地93个1→0错题。老师平台可能只显示两位小数，单次0.01差值需谨慎解释。

## 对接下来工作的影响

1. 将0.54作为首要平台对照，原0.49保留用于定位改动，不再把未测新训练版当作比0.54更优。
2. 优先检查“哪些样本进入fallback、6种排列各自的分歧和互补”，再考虑融合权重或路由修改；应以整记录/房间验证约束，避免继续堆输出偏置。
3. 微调模型仍可以做独立对照，但需要重新评估它是否保留这套12视角优势。不能把未测微调与0.54直接相加就假定更强。
4. 暂不沿C1房间偏置和NT频谱斜率均衡继续加码；两者在这次平台反馈中下降明显。Gcal与GNP作为互补对照保留，不能凭相同0.53就认为预测相同。

本次只登记、结构核对、少量公式运行验证，没有训练或生成新候选，没有推送GitHub。
'''
    (ROOT/'docs/TEAMMATE054_20261025.md').write_text(report,encoding='utf-8')
    for filename,link in [('README.md','docs/TEAMMATE054_20261025.md'),('docs/SUMMARY.md','TEAMMATE054_20261025.md'),('models/README.md','../docs/TEAMMATE054_20261025.md')]:
        p=ROOT/filename;text=p.read_text(encoding='utf-8');heading,rest=text.split('\n',1)
        notice=f'当前团队最高平台成绩已到 **0.54**：`{records[0]["path"]}`'
        if notice not in rest:p.write_text(heading+f'\n\n{notice}（83.68MB）。本次登记5份队友模型，确认主要提升来自保留0.49保护规则后的12视角fallback融合，原神经网络权重不变。[五模型分析]({link})。原微调候选仍未评分；以下旧状态为历史记录。\n'+rest,encoding='utf-8')
    print('COMPLETE',records[0]['path'],flush=True)

if __name__=='__main__':main()
