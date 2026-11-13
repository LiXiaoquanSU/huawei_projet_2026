"""Record the verified exploratory artifact in team documents, not platform scores."""
import shutil
from run import *

def main():
    meta=read(WORK/'published.json');assert meta['status']=='unscored_exploratory'
    s=meta['selection'];assert s['group']=='rx2'
    resource=meta['resource_check'];assert resource['status']=='pass'
    maxmean=max(c['cpu_mean_s'] for c in resource['cases']);peak=resource['peak_working_set_bytes']
    rr=read(WORK/'reduced_results.json');rr=[r for r in rr if r['method']=='adjacent_up_rx2' and r['room']!='actual']
    base=sum(r['baseline']['correct'] for r in rr);new=sum(r['metrics']['correct'] for r in rr);total=sum(r['metrics']['total'] for r in rr)
    corrected=sum(r['metrics']['corrected'] for r in rr);broken=sum(r['metrics']['broken'] for r in rr)
    tail=f'''\n## 本轮交付：仅作平台探索，不替换已测0.54\n
- 文件：`{meta['path']}`；平台未测，不能称为0.60或已证实升级。
- 大小{meta['bytes']/1e6:.2f} MB；独立CPU4线程最慢案例平均{maxmean:.4f}秒，峰值工作集{peak/1e6:.1f}MB；结构、接口、嵌入权重、真实输入及零输入检查通过。机型不同，耗时仍可能变化。
- 主模型是原0.54十二视角，不改变其权重和保护逻辑。第二模型来自队友seed310，只做原输入与时间反转两个视角，并单独校准。75%主模型概率+25%专家概率仅在融合argmax为原argmax+1时采用，否则返回原主模型的原始分数。
- 四个留房间开发对照：{base}/{total}→{new}/{total}，纠错{corrected}、破坏{broken}，净增{new-base}个，提升{s['pooled_gain']*100:.2f}个百分点。a1 +1.25、c1 +5.00、g1 +0.29、e1 0.00个百分点。
- 来源训练数据诊断：{s['source_diagnostic']['correct']}/960={s['source_diagnostic']['accuracy']*100:.2f}%；并非独立测试或老师测试成绩。
- 正式门槛要求至少+2个百分点，本候选未达到；没有改变门槛或把它登记为通过。保存为探索候选是因为三个房间提升、一个不变，整记录bootstrap区间约+0.23～+3.86个百分点，且比双十二视角版明显省时。该区间未校正多次选择，开发房间也已重复使用，不是泛化保证。
- 全量双十二视角版本本地约0.80秒，未通过耗时检查，不交付为可提交候选。2/4/6专家视角比较及全部失败记录均保留。
- 当前已测最好仍为0.54；registry的next_submission和已测最佳不因本探索文件而替换。下一步价值由平台实测决定，不能从本地+1.82推算平台一定增加同样幅度。
- 模型SHA256：`{meta['sha256']}`。
'''
    p=ROOT/'docs/SEED_EXPERTS_20261025.md';text=p.read_text(encoding='utf-8').split('\n## 本轮交付：')[0]+tail
    p.write_text(text,encoding='utf-8');(PACKAGE/'下一轮实验_跨种子条件专家.md').write_text(text,encoding='utf-8')
    # Preserve original score columns and seed statistics; local results stay separate.
    csvpath=PACKAGE/'新实验_留房间结果.csv'
    with csvpath.open(encoding='utf-8-sig',newline='') as f:existing=list(csv.DictReader(f));fields=list(existing[0])
    reduced=read(WORK/'reduced_results.json');methods={r['method'] for r in reduced};existing=[r for r in existing if r['method'] not in methods]
    for r in reduced:
        m=r['metrics'];b=r['baseline'];existing.append(dict(room=r['room'],seed=r['seed'],method=r['method'],
          split='训练来源诊断' if r['room']=='actual' else '历史开发留房间',accuracy=m['accuracy'],baseline_accuracy=b['accuracy'],
          corrected=m['corrected'],broken=m['broken'],changed=m['changed'],windows=m['total'],platform_score='未测试'))
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(existing)
    full=ROOT/'docs/TEAM_PROGRESS_FULL_20261025.md';content=full.read_text(encoding='utf-8').split('\n第三阶段及交付：')[0]
    content+=f'\n第三阶段及交付：双十二视角超时后，比较第二专家2/4/6视角，总计8次训练、19种组合。保留2视角探索候选，留房间{base}/{total}→{new}/{total}，+{s["pooled_gain"]*100:.2f}个百分点；独立CPU与数值检查通过，但未达到预设+2个百分点正式门槛，平台未测。原0.54最佳保持。详情和准确资源数据见docs/SEED_EXPERTS_20261025.md。\n'
    full.write_text(content,encoding='utf-8');shutil.copy2(full,PACKAGE/'团队完整总结_含小全实验.md')
    target=PACKAGE/'新候选模型';target.mkdir(exist_ok=True);src=ROOT/meta['path'];dest=target/src.name
    if dest.exists():assert sha(dest)==meta['sha256']
    else:shutil.copy2(src,dest)
    save(dest.with_suffix('.json'),meta);assert sha(dest)==meta['sha256']
    (target/'先读说明.md').write_text('本目录只有未测探索候选，不是已测0.54最佳或已到0.60的模型。先看上级目录下一轮实验_跨种子条件专家.md。当前文件的本地+1.82个百分点未达到预设+2个百分点正式门槛，已通过本地资源与数值检查。老师平台效果未知。\n',encoding='utf-8')
    readme=PACKAGE/'README.md';text=readme.read_text(encoding='utf-8').split('\n## 本轮实验进度\n')[0]
    text+=f'\n## 本轮实验进度\n\n已完成8次留房间训练、19种组合；详细见`下一轮实验_跨种子条件专家.md`和`新实验_留房间结果.csv`。第二专家2视角方案提升{new-base}/{total}个（+{s["pooled_gain"]*100:.2f}个百分点），已通过资源/数值检查，但未达到预设+2个百分点门槛，仅作探索。模型放在`新候选模型/`，平台未测；已测最佳仍0.54。\n'
    readme.write_text(text,encoding='utf-8')
    with (PACKAGE/'新增模型清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['文件','平台分','状态','sha256']);w.writeheader()
        w.writerow({'文件':'original_run/final.pt','平台分':'0.37','状态':'原始检查点，不能当成完整054推理图','sha256':sha(PACKAGE/'original_run/final.pt')})
        w.writerow({'文件':'新候选模型/'+src.name,'平台分':'未测试','状态':'探索候选，未通过正式提升门槛','sha256':meta['sha256']})
    print(json.dumps(dict(path=meta['path'],bytes=meta['bytes'],cpu_max_mean=maxmean,local_delta=new-base,local_total=total,
                         source_accuracy=s['source_diagnostic']['accuracy'],platform_score=None,package_copy=str(dest)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
