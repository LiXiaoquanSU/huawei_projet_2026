"""Audit cross-fitting boundaries and summarize conditional protection research."""
from learn import *

def main():
    results=read(WORK/'results.json');selection=read(WORK/'selection.json');proof=[]
    for room in (*ROOMS,'actual'):
        paths=sorted((Path('D:/huawei2026-experiments/protection_arbiter_20261025')/'crossfit'/room).glob('fold*/predictions.pt'))
        if not paths:continue
        assert len(paths)==3
        seen=set()
        for path in paths:
            p=torch.load(path,weights_only=True)
            fit=set(p['fit_records']);cal=set(p['calibration_records']);test=set(p['oof_records'])
            assert not fit&cal and not fit&test and not cal&test and not seen&test
            assert {r['identifier'] for r in p['rows']}==test
            if room!='actual':assert all(r['room']!=room for r in p['rows'])
            seen|=test
            m=read(path.parent/'manifest.json')
            assert sha(path.parent/'final.pt')==m['checkpoint_sha256']
            proof.append(dict(room=room,fold=m['fold'],checkpoint_sha256=m['checkpoint_sha256'],fit_records=sorted(fit),calibration_records=sorted(cal),oof_records=sorted(test)))
    save(WORK/'crossfit_audit.json',dict(checkpoints=len(proof),passed=True,folds=proof,
        feature_cache={n:sha(PREV/n) for n in ['rows.json','s.npy','t.npy']}))
    lines=['# 条件保护仲裁器：独立分折纠错研究','','平台最佳仍0.54。本轮原底座和人数偏置不变，新训练小仲裁器，研究何时改用其他视角的判断。',
      '', '## 数据与训练边界', '',
      '- 每个外层目标房间完全排除；对应源fit记录再分3折，源calibration记录单独留出。12份新底座都从随机种子42初始化，AdamW、5轮、batch24、lr0.0004、weight_decay0.01。实际FLOAT32 ONNX预处理缓存，与原run训练的FLOAT64路径不同，不声称精确复现0.37训练。',
      '- 新底座仅用于生成纠错样本，不替换提交用原权重。每条纠错样本的CSI和标签均不进入产生该预测的底座训练或偏置/温度校准。',
      '- 20种意见含原054、完整12视角融合、6种Rx双时间、12个单视角。特征含置信度、间隔、熵、支持票、分歧；分别测试线性、小关系MLP、带候选人数编码MLP。',
      '- 仲裁器训练效用为救回+1、改坏−2。两个种子42/2026；固定300步AdamW。源OOF再次按记录折验证，仅在源记录上选group/views与阈值0/.1/.2/.35/.5/不改判。',
      '- 源元验证是重复开发的OOF堆叠，不是全新盲测。外层房间的所有标签均未进入底座、校准、仲裁器或阈值选型；外层标签在全部训练/源选型完成后才用于本报告。',
      '- a1/c1/g1/e1为已反复开发的房间，指标不等于老师隐藏测试准确率。没有假定每个房间人数均匀，没有使用官方U标签或外部数据。',
      '', '## 外层结果', '',
      '|方法|正确/880|救回|改坏|净增|a1百分点|c1百分点|g1百分点|e1百分点|按记录95%提升区间|过门槛|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in results:
        correct=sum(x['correct'] for x in r['records']);rescued=sum(x['metrics']['corrected'] for x in r['rooms'].values());broken=sum(x['metrics']['broken'] for x in r['rooms'].values())
        lo,hi=r['ci95'];g=[f'{100*r["room_gains"][room]:+.2f}' for room in ROOMS]
        lines.append('|'+ '|'.join([r['method'],f'{correct}/880',str(rescued),str(broken),str(r['net']),*g,f'{100*lo:+.3f}～{100*hi:+.3f}',str(r['passed'])])+'|')
    lines+=['','对照278/880。门槛：整体至少+2个百分点，至少两房间提升，无房间低于−2个百分点，按整记录重采样的区间下界>0。source_selected为预设主方法；其余为固定特征方案的探索性对照，不能隐藏多方案开发选择。区间以固定四房间内记录为重采样单位，不反映新房间总体的不确定性。',
       '', '## 选型与交付', '',f'通过门槛后选用：{selection["selected"] or "无"}。平台成绩尚未新增。']
    if (WORK/'actual_metrics.json').exists():
        a=read(WORK/'actual_metrics.json');lines += ['',f'实际原权重来源数据：{a["baseline"]["correct"]}/960 → {a["metrics"]["correct"]}/960；改判{a["changes"]}窗。全部48记录训练过原底座，因此不是独立测试。',
           '实际仲裁器由另3份随机初始化底座生成的全量记录OOF样本训练；5条独立记录只用于这些OOF底座的偏置/温度校准，其余43条用于生成仲裁器训练样本。提交底座仍保留原48条训练的原权重。']
    if (WORK/'published.json').exists():
        m=read(WORK/'published.json');check=m['resource_check'];worst=max(x['cpu_mean_s'] for x in check['cases'])
        lines += ['',f'新文件：{m["path"]}，{m["bytes"]/1e6:.2f}MB；SHA256 {m["sha256"]}。',
                  f'实际ONNX与PyTorch/缓存对照通过；CPU4线程所测用例中最慢平均{worst:.3f}秒，大小/时间/内存限制核验通过。平台未知，不能称已超过0.54。']
    else:lines+=['','尚未发布推荐ONNX。训练产物保留供审计，不将未过门槛模型包装为升级。']
    if AUX_HEADS:
        lines+=['','## 内部专家扩展','','最终方案还加入6种原网络内部意见：直接分类、有序人数、静态/动态/相位/时间专家，共26意见。共享编码器，不做6次额外完整推理；各意见使用当前教师原偏置/温度，没有单独拟合额外头偏置。小仲裁器包含意见类型编码，用来学习不同分支的可靠性。',
                '保护范围外，只允许新增内部意见参与0/1之间纠错；不能用单视角把原融合的正确1人任意改为2/3/4人。保护范围内按仲裁器决策。原显式人数偏置不变，但学习路由仍可能改变有效的类别分布。']
    if ALIGNED:
        lines+=['','## 分头校准与导出拦截','','原未校准内部专家在留房间开发上通过过门槛，但迁移到真实原权重后来源数据676→373，已拦截、未发布。最终融合偏置不能直接当作所有内部专家的合适校准。',
                '单独校准内部头；原054最终偏置保持原值。头校准参数来自源calibration记录，外层目标房间完全排除。']
    if LABEL_CAL:
        lines+=['','本版采用源校准记录的真实人数比例，逐头拟合加性校准，使这些源校准记录上的平均预测匹配观测人数比例（固定400步）。这是固定logit条件下源校准交叉熵的类别偏置拟合，不是强制所有目标房间均匀。每折校准集按设计每类一条20窗记录；这些记录不训练对应OOF底座。',
                '此前用教师预测边际校准也做过对照，但教师在校准记录上会出错，不能把其预测边际当作真实人数比例。',
                '本版从仲裁训练开始就只处理双时间0/2保护窗口；原视角1人及所有其他fallback窗口保留。最终导出还有来源数据最多下降29/960的崩溃拦截；来源数据已训练原底座，不能当作独立测试。']
    lines+=['','## 同轮失败与中间对照','','以下为迭代开发记录，不能把多轮比较当成一次新的盲测。','','|方案|源选型流程净增/880|固定关系MLP净增/880|','|---|---:|---:|']
    basepath=Path('D:/huawei2026-experiments/protection_arbiter_20261025')
    for dirname,label in [('', '全窗口硬仲裁'),('guard_only','仅保护窗口硬仲裁'),('pair_one','只解除双时间0/2且改为1'),('guard_soft','保护窗口软融合')]:
        path=basepath/dirname/'results.json'
        if path.exists():
            rr=read(path);a=next(r for r in rr if r['method']=='source_selected');b=next(r for r in rr if r['method']=='relation_mlp')
            lines.append(f'|{label}|{a["net"]:+d}|{b["net"]:+d}|')
    lines+=['','## 记录与复现','','代码：experiments/protection_arbiter_20261025/README.md。权重、OOF意见、每条记录清单、参数、验证输出及校验在D:/huawei2026-experiments/protection_arbiter_20261025/。未推送Git或上传老师平台。']
    text='\n'.join(lines)+'\n';(ROOT/'docs/PROTECTION_ARBITER_20261025.md').write_text(text,encoding='utf-8')
    (ROOT/'2026-10-25_037配方与种子复现/条件保护仲裁器研究.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(checkpoints=len(proof),selected=selection['selected'],report='docs/PROTECTION_ARBITER_20261025.md'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
