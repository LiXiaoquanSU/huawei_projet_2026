import importlib.util,shutil
from run import *

def main():
    sys.path.insert(0,str(ROOT/'experiments/protection_arbiter_20261025'))
    results=old.read(WORK/'results.json');rows=old.read(CACHE/'rows.json');detail={};params=sum(p.numel() for p in CovarianceNet().parameters())
    spec=importlib.util.spec_from_file_location('covariance_bootstrap',ROOT/'experiments/protection_arbiter_20261025/evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    evid=Path('D:/huawei2026-experiments/subnet_expectation_20261025/evidence')
    for seed in (42,2026):
      records=[];nlls=[]
      for room in ROOMS:
        predictions=[]
        for part in (0,1):
            folder=WORK/f'{room}_p{part}_s{seed}';r=old.read(folder/'result.json');cp=torch.load(folder/'final.pt',weights_only=True)
            assert old.sha(folder/'final.pt')==r['checkpoint_sha256'];assert not set(cp['fit_records'])&set(cp['calibration_records']);assert not set(cp['fit_records']+cp['calibration_records'])&set(cp['test_records'])
            entry=r['history'][r['epoch']-1];nlls.append(entry['calibration_nll'])
            detail[folder.name]=dict(epoch=r['epoch'],unrestricted_epoch=r['unrestricted_selection']['epoch'],fit=entry['fit'],calibration=entry['calibration'],test=r['test'],last_fit=r['history'][-1]['fit'])
            d=torch.load(folder/'predictions.pt',weights_only=True);predictions.append(d['logits'].exp())
        z=torch.stack(predictions).mean(0);y=d['labels'];ix=d['test_indices'];ref=torch.load(evid/f'{room}.pt',weights_only=True);b,T=old.calibration(room);base=old.policy(ref['raw'],b,T).argmax(-1);p=z.argmax(-1)
        for ident in sorted({rows[i]['identifier'] for i in ix}):
            jj=[i for i in ix if rows[i]['identifier']==ident];delta=int((p[jj]==y[jj]).sum()-(base[jj]==y[jj]).sum());records.append(dict(room=room,identifier=ident,windows=len(jj),delta=delta))
      results[f's{seed}_pmean']['bootstrap_ci_vs054proxy']=ev.bootstrap(records);results[f's{seed}_pmean']['mean_source_calibration_nll']=float(np.mean(nlls))
    old.save(WORK/'audit.json',dict(parameters=params,results=results,individual=detail,checkpoint_hashes_verified=16,platform_score=None))
    lines=['# 归一化时间协方差新结构：2026-10-26','',f'完成16个神经网络训练，每个40轮，共640轮；每个网络{params:,}个参数。平台未测试，最佳已评分仍0.54。','',
    '## 实验设计','',
    '输入为幅度/相位对的时间协方差图及差分协方差图，按接收通道或相位对池化均值和标准差，共8张30×30图。去除时间均值并归一化能量，再以三层卷积和全局统计池化预测0—4人。无旧权重、无绝对幅度分支、无外部数据、无隐藏标签。接收通道排列数值一致性已检查。',
    '', '四个留房间折×两个整记录校准划分×两个种子。每套校准只含源房间五条记录；测试房间不进入训练和校准。按人数×房间加权采样，AdamW，时间反转一致性损失。40轮内按源校准交叉熵选权重，但至少训练5轮；同时记录不加最少轮数限制时的选择。这个5轮限制只是防止过早停止的预设启发式，并不保证学充分。',
    '', '两套校准训练有重叠，不是独立数据扩增。四个房间已反复作为开发集使用，不应将结果视作新的盲测；旧054代理也不是实际平台054原权重。',
    '', '## 本地结果','', '|种子|校准划分|正确/880|准确率|','|---|---|---:|---:|']
    for k,r in results.items():lines.append(f"|{k.split('_')[0]}|{k.split('_')[1]}|{r['correct']}/880|{r['correct']/880:.2%}|")
    lines+=['','对照：旧054代理278/880（31.59%）；上一轮强度新结构固定分层301/880（34.20%）。mean表示两套校准模型概率平均，不是额外训练。','', '## 按房间与类别检查','']
    for seed in (42,2026):
        r=results[f's{seed}_pmean'];cm=np.array(r['confusion']);lines+=[f"- 种子{seed}两套平均：各房间"+'、'.join(f"{v['room']} {v['correct']}/{v['total']}" for v in r['rooms'])+f"；0人{cm[0,0]}/{cm[0].sum()}，4人{cm[4,4]}/{cm[4].sum()}。",f"  相对旧054代理的整记录分层bootstrap 95%差值区间：{r['bootstrap_ci_vs054proxy']}。此区间不包含搜索模型和选种子的额外不确定性。"]
    lines+=['','## 训练充分性与校准稳定性','', '|模型|选中轮数|无下限轮数|源训练正确/总数|源校准正确/总数|','|---|---:|---:|---:|---:|']
    for name,r in detail.items():lines.append(f"|{name}|{r['epoch']}|{r['unrestricted_epoch']}|{r['fit']['correct']}/{r['fit']['total']}|{r['calibration']['correct']}/{r['calibration']['total']}|")
    passed=any(r['correct']>301 and r.get('bootstrap_ci_vs054proxy',[0])[0]>0 and all(v['correct']>=b for v,b in zip(r['rooms'],[45,40,174,19])) for k,r in results.items() if 'mean' in k)
    lines+=['','## 结论','',('存在通过初步代理门槛的组合，还需全数据训练与实际原模型错题/运行资源验证，暂不推荐提交。' if passed else '本轮没有通过代理对照门槛，不导出推荐提交模型。归一化和不变性本身不保证迁移：去除背景也可能丢失静态人数信息。应保留这次阴性结果，不能通过挑某个房间或某个种子局部提高来宣布成功。'),'',
    '后续若继续结构实验，应检验保留静态信息的分支与协方差分支是否互补，并把固定训练预算对照与校准选择对照分开。整记录校准的敏感性仍需更多划分验证。仅凭本轮不能断言静态信息就是失败原因。','',
    '## 文件','', '`experiments/covariance_20261026/` 保存特征、训练和报告代码；`D:/huawei2026-experiments/covariance_20261026/` 保存16份权重、预测、每轮源混淆矩阵、特征检查和结果。未修改平台054文件。']
    path=ROOT/'docs/COVARIANCE_20261026.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(path,ROOT/'2026-10-25_037配方与种子复现/归一化协方差新结构_20261026.md')
    reg=old.read(ROOT/'models/registry.json');reg['research_status']='covariance_completed';reg['research_report']='docs/COVARIANCE_20261026.md';reg['covariance_20261026']=dict(fits=16,results=results,platform_score=None,preliminary_gate_passed=passed);old.save(ROOT/'models/registry.json',reg)
    print(json.dumps(dict(parameters=params,preliminary_gate_passed=passed,results=results)),flush=True)

if __name__=='__main__':main()
