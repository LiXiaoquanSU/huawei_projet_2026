"""Final delivery report; retain failed hypotheses and the source tradeoff."""
from bayes import *

def main():
    published=read(WORK/'published.json');r=published['development'];check=published['resource_check'];source=published['source_metrics']
    rows=read(PREV/'rows.json');recordroom={x['identifier']:x['room'] for x in rows};proof=[]
    for room in (*ROOMS,'actual'):
        for k in range(3):
            folder=WORK.parent/'crossfit'/room/f'fold{k}';m=read(folder/'manifest.json')
            a,b,c=[set(m[key]) for key in ['fit_records','calibration_records','oof_records']]
            assert not a&b and not a&c and not b&c
            if room!='actual':assert all(recordroom[x]!=room for x in a|b|c)
            assert sha(folder/'final.pt')==m['checkpoint_sha256'];proof.append(m)
    save(WORK/'training_audit.json',dict(passed=True,nested_models=15,manifests=proof,
      feature_cache_sha256={n:sha(PREV/n) for n in ['rows.json','s.npy','t.npy']},original_backbone_sha256=sha(ROOT/'models/best_037/final.pt')))
    corrected=sum(x['corrected'] for x in r['rooms'].values());broken=sum(x['broken'] for x in r['rooms'].values())
    lines=['# 条件纠错研究与下一份待测模型','','已交付：`'+published['path']+'`。平台未知，不能称已超过0.54或达到0.60。',
      '', '## 最终结果', '',
      f'- 四留房间代理：278/880 → {278+r["net"]}/880，净增{r["net"]}窗（+{100*r["net"]/880:.2f}个百分点）；救回{corrected}窗、改坏{broken}窗。',
      f'- 固定四房间内按整记录重采样的95%提升区间：+{100*r["ci95"][0]:.2f}～+{100*r["ci95"][1]:.2f}个百分点。房间/记录已反复用于开发，此区间不涵盖新房间总体或多轮选型不确定性。',
      f'- 原底座训练来源：676/960 → {source["correct"]}/960；救回{source["corrected"]}、改坏{source["broken"]}。下降29窗是明确风险，不把本轮写成所有集合都提升。来源数据参与过底座训练，不是独立测试。',
      '', '|留出房间|原正确|新正确|净增|源选定阈值|源选定融合比例|','|---|---:|---:|---:|---:|---:|']
    for room in ROOMS:
        v=r['rooms'][room];delta=sum(x['delta'] for x in r['records'] if x['room']==room);cp=torch.load(WORK/f'{room}.pt',weights_only=True)
        lines.append(f'|{room}|{v["correct"]-delta}|{v["correct"]}|{delta:+d}|{cp["config"]["threshold"]}|{cp["config"]["amount"]}|')
    actual=torch.load(WORK/'actual.pt',weights_only=True)['config']
    lines += ['', '## 这次真正改变的部分', '',
      '提交底座仍是原0.37四路线权重及原054校准/保护体系；没有重新初始化提交底座，没有新增最终人数偏置，也没有采用失败内部专家方案的头偏置。',
      '原视角1人优先和其他fallback结果保留。只在双时间0/2保护分支内，比较原判断与其他视角判断的相邻人数候选。',
      '仲裁输入使用未加类别偏置的原始视角对数赔率、其跨视角变化、支持票与熵。比较的是同一候选人数相对原人数的证据；不是把不同随机种子的校准后置信度直接当作可比量，也不是声称完整模型对先验不变。',
      '候选还须在原始分数中比原人数更有支持。两个12→16→1小MLP，每个225参数，共450参数。它们只控制纠错，不新增完整编码器或物理活动标签。',
      f'实际提交文件使用源OOF选出的阈值{actual["threshold"]}、融合比例{actual["amount"]}；这些参数没有用老师隐藏测试集拟合。',
      '', '## 训练与防泄漏', '',
      '- 12份外层嵌套底座：a1/c1/g1/e1分别完全排除；源fit按采集记录分3折，底座训练2折，另1折生成纠错样本；5条源calibration记录独立校准该底座。每条纠错样本都不参与产生其预测的底座训练或校准。',
      '- 另3份完整记录的分折底座供最终仲裁器生成OOF样本。提交的原底座仍训练过全部48条；43条记录生成仲裁训练样本，5条独立记录用于这些新OOF底座的校准。',
      '- 新底座随机种子42、5轮AdamW、batch24、lr0.0004、weight_decay0.01，实际FLOAT32 ONNX预处理缓存；不是原0.37 run的逐位复现。',
      '- 最终小仲裁器种子42/2026，固定300步AdamW、lr0.005、weight_decay0.1；救回收益1、误改代价4。源OOF再按记录3折选择阈值0/.1/.25/.5与融合.25/.5/1/保持原值，以源类别×房间平衡对数损失选型。外层标签只在源训练/选择完成后用于评估。',
      '- 代理沿用历史划分，排除unknown源记录；b1仅一条记录，不作为外层房间。原始提交底座仍含全部48记录。没有外部数据、官方U训练、隐藏测试标签、猜测活动标签或新增滑窗。',
      '', '## 本轮失败对照', '',
      '以下为迭代研究，不把多次使用同一开发房间当作新的盲测。', '',
      '|实验目录|源自动选型流程净增/880|固定计数MLP净增/880|','|---|---:|---:|']
    for name in ['', 'guard_only','pair_one','guard_soft','aux_heads/gates','aux_heads/gates/keep_one','aux_aligned/gates','aux_aligned/gates/strict_scope','aux_labelcal/gates/evidence_guard','aux_labelcal/gates/cost4']:
        path=WORK.parent/name/'results.json'
        if not path.exists():continue
        rr=read(path)
        if not isinstance(rr,list):continue
        a=next(x for x in rr if x['method']=='source_selected');c=next(x for x in rr if x['method']=='count_mlp')
        lines.append(f'|{name or "全窗口仲裁"}|{a["net"]:+d}|{c["net"]:+d}|')
    lines += ['', '未校准内部专家版曾在开发上净增32窗，但实际原权重来源数据676→373，已阻断发布。逐头匹配教师预测边际、逐头用源真标签校准、只解除部分保护、提高误改代价等均保留完整对照；没有把这些失败的ONNX混入待提交目录。',
      '', '## 提交文件与核验', '',
      f'- 文件：{published["path"]}；{published["bytes"]/1e6:.2f}MB；SHA256 `{published["sha256"]}`。',
      '- ONNX单文件嵌入权重，frames float32[600,1,3,1024,2]、intervals float32[600]，输出float32[5]。',
      f'- 7个真实CSI窗口及全零输入：PyTorch/缓存/ONNX数值和分类核对通过；最大概率误差{max(x["probability_max_error"] for x in check["cases"]):.2g}。',
      f'- 本机CPU4线程，8用例中最慢平均{max(x["cpu_mean_s"] for x in check["cases"]):.3f}秒；峰值提交内存{check["peak_commit_bytes"]/1e9:.2f}GB。大小<100MB、平均<0.7秒、内存<6GB均通过本机核验，不保证老师机器等速。',
      '', '## 实际建议与边界', '',
      '这是下一份待平台验证的实验候选；本地留房间通过本轮门槛，但多轮开发选型和来源正确数下降都意味着平台可能不升反降。保留已评分0.54为当前最佳。不能把0.54与本地+2.27个百分点相加，推断网站得分。',
      '原始对数赔率仍是判别网络分数构成的启发式证据，不是已证明的物理生成似然或房间人数真值。',
      '', '训练权重/记录清单/原始输出/校验：D:/huawei2026-experiments/protection_arbiter_20261025/。最终分支：prior_free/。代码：experiments/protection_arbiter_20261025/bayes.py与export_bayes.py。未上传老师平台、未推送Git。']
    text='\n'.join(lines)+'\n';(ROOT/'docs/PROTECTION_ARBITER_20261025.md').write_text(text,encoding='utf-8')
    package=ROOT/'2026-10-25_037配方与种子复现';(package/'条件保护仲裁器研究.md').write_text(text,encoding='utf-8')
    status='最新待测：01_prior_free_evidence_arbiter.onnx（41.95MB）。留房间278→298/880，来源676→647/960，平台未知，最佳仍0.54。见docs/PROTECTION_ARBITER_20261025.md。'
    for name in ['README.md','models/README.md','docs/TEAM_PROGRESS_FULL_20261025.md']:
        path=ROOT/name;oldtext=path.read_text(encoding='utf-8')
        if status not in oldtext:
            first,rest=oldtext.split('\n',1);path.write_text(first+'\n\n'+status+'\n\n'+rest.lstrip('\n'),encoding='utf-8')
    (package/'团队完整总结_含小全实验.md').write_text((ROOT/'docs/TEAM_PROGRESS_FULL_20261025.md').read_text(encoding='utf-8'),encoding='utf-8')
    print('Report and15nested checkpoint audits complete',flush=True)

if __name__=='__main__':main()
