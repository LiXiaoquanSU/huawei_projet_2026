"""Audit whole-record uncertainty and document both new directions."""
from run import *

PACKAGE=ROOT/'2026-10-25_037配方与种子复现'

def uncertainty(item):
    groups=[]
    for room in ROOMS[:-1]:
        d=torch.load(WORK/f'pred_{room}_{item["key"]}.pt',weights_only=True)
        rows=read(PREV/'rows.json')
        for record in sorted({rows[i]['identifier'] for i in d['indices']}):
            ix=[i for i in d['indices'] if rows[i]['identifier']==record];y=d['labels'][ix]
            groups.append(dict(room=room,identifier=record,n=len(ix),
                delta=int((d['logits'][ix].argmax(-1)==y).sum()-(d['baseline'][ix].argmax(-1)==y).sum())))
    rng=np.random.default_rng(20261025);values=[]
    for _ in range(5000):
        selected=[]
        for room in ROOMS[:-1]:
            rr=[r for r in groups if r['room']==room];selected.extend(rr[i] for i in rng.integers(0,len(rr),size=len(rr)))
        values.append(sum(r['delta'] for r in selected)/sum(r['n'] for r in selected))
    return dict(ci95=[float(x) for x in np.percentile(values,[2.5,97.5])],records=groups,
                probability_nonpositive=float(np.mean(np.array(values)<=0)),note='Reused development; not selection-adjusted or hidden-test uncertainty.')

def main():
    rr=read(WORK/'results.json');clockresults=read(clock.WORK/'results.json')
    for r in rr:
        r['bootstrap']=uncertainty(r);r['robust_pass']=r['eligible'] and r['bootstrap']['ci95'][0]>0
    accepted=[r for r in rr if r['robust_pass']]
    selected=accepted[0] if accepted else None
    save(WORK/'final_results.json',rr);save(WORK/'selection.json',dict(selected=selected,platform_score=None,
      reason='Fixed stage1 candidates, followed by2 documented lowcount controls; >2pp,>=2positive rooms,no>2pploss,whole-recordCIpositive. Source not criterion.'))
    lines=['# 0.53之后：时间原点与训练子网络实验','','上轮跨种子相邻向上候选平台0.53，低于原0.54，按精确文件SHA已登记。目标仍0.60，本轮没有隐藏标签或平台结果。',
      '', '## 方法与对照', '',
      '- 时间原点：六个固定正负时钟偏移，三个偏移对×四种融合，共12种方案；不修改真实CSI/ATI。零偏移控制五模型均完全一致。修正了初始fallback温度与池化顺序，最终结果用修正实现统一重算。最好只净增1/880，未导出。',
      '- 路线子网络：四个路线训练时各有25%丢弃概率，枚举15个有效mask，按原代码全丢恢复规则计算真实概率。关闭普通Dropout/MixStyle，只研究路线dropout的期望；不声称是全部训练随机性的精确边缘化。',
      '- 原训练的gate先看完整内容再按mask归一，shared分类器使用mask后的内容，presence依然看完整动态/相位/时间证据。评估严格复现这个顺序。full-mask与原输出的最大logit误差≤4.8e-7。',
      '- 比较15子网期望、仅丢静态、仅丢时间三种读出×四种融合，共12种固定方案；再按历史选择性054经验加入两个保护高人数的对照，共14种。所有权重冻结，没有新偏置拟合、外部数据或平台标签。',
      '- 使用原留房间模型；拟合和校准未见目标房间，按整条记录统计。但这些房间已反复用于历史开发，bootstrap也未校正多次选择，不是新盲测。',
      '', '## 全部子网络比较', '',
      '|方案|净纠错/880|a1提升|c1提升|g1提升|e1提升|整记录95%区间|通过|',
      '|---|---:|---:|---:|---:|---:|---|---|']
    for r in rr:
        lo,hi=r['bootstrap']['ci95'];g=[f'{r["room_gains"][room]*100:+.2f}' for room in ROOMS[:-1]]
        lines.append('|'+ '|'.join([r['key'],str(r['proxy_net']),*g,f'{lo*100:+.2f}～{hi*100:+.2f}',str(r['robust_pass'])])+'|')
    if selected:
        source=selected['rooms'][-1]
        lines += ['','## 本轮选择', '',f'选择`{selected["key"]}`，四个开发房间均改善，净增{selected["proxy_net"]}/{selected["proxy_total"]}，提升{selected["pooled_gain"]*100:.2f}个百分点。',
          f'来源训练诊断反而{source["baseline"]["correct"]}/960→{source["metrics"]["correct"]}/960，不用来源高分冒充泛化，不隐去这个退步。',
          '物理解释是候选假设：时序表示仍用于presence；减少其在具体人数分类的独立专家/共享输入中的作用。不能称为从模型中完全删除时间信息，也不能据此证明时间信号没有用。',
          '使用同一组0.37基础权重，原模型与“训练时丢时间路线”读出各形成完整保护/视角融合结果，再按概率融合。新结构是共享特征的双人数读出，非第二独立种子、非直接改偏置。',
          '需要导出后的数值、时间、内存检查；平台是否超过0.54未知。']
    else:lines+=['','本轮没有通过门槛的候选，不把失败对照包装成升级。']
    report='\n'.join(lines)+'\n';(ROOT/'docs/SUBNET_EXPECTATION_20261025.md').write_text(report,encoding='utf-8')
    (PACKAGE/'后续实验_时间原点与子网络.md').write_text(report,encoding='utf-8')
    with (PACKAGE/'后续实验_子网络结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=__import__('csv').DictWriter(f,fieldnames=['方案','房间','正确','总窗数','对照正确','纠错','破坏','平台分']);w.writeheader()
        for r in rr:
            for item in r['rooms']:
                m=item['metrics'];w.writerow({'方案':r['key'],'房间':item['room'],'正确':m['correct'],'总窗数':m['total'],
                  '对照正确':item['baseline']['correct'],'纠错':m['corrected'],'破坏':m['broken'],'平台分':'未测试'})
    print(json.dumps(dict(selected=selected['key'] if selected else None,proxy_net=selected['proxy_net'] if selected else None,
                         ci=selected['bootstrap']['ci95'] if selected else None),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
