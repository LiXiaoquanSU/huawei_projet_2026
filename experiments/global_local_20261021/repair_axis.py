"""Preserve invalid startup output and check segment identity before restart."""
import shutil
from pathlib import Path
import torch
from network import ROOT,build,event_stats
WORK=Path('D:/huawei2026-experiments/global_local_20261021')
def main():
    invalid=WORK/'_invalid_segment_axis_v0';invalid.mkdir(exist_ok=True)
    source=(WORK/'fits').resolve();target=(invalid/'fits').resolve();assert source.parent==WORK.resolve() and target.parent.parent==WORK.resolve()
    if source.exists():
        assert not target.exists();source.rename(target)
    prior=WORK/'source_snapshot/network.py'
    if prior.exists():shutil.copy2(prior,invalid/'network_before_fix.py')
    (invalid/'README.md').write_text('启动复查发现event_stats将不同片段而非同片段统计拼接，停止训练并完整保留此目录。该版本不用于选择和最终指标。修正为同一片段内拼接后，全部新结构重跑；未用这些错误结果挑选结构。\n',encoding='utf-8')
    # Distinct constant segments must preserve segment identity and zero changes.
    e=torch.arange(1,6,dtype=torch.float32).reshape(1,5,1,1).expand(2,5,32,30);stats=event_stats(e)
    assert stats.shape==(2,5,128) and torch.allclose(stats[:,:,:32],e[:,:,:,0]) and torch.allclose(stats[:,:,64:96],e[:,:,:,0]) and stats[:,:,96:].abs().max()==0
    torch.set_num_threads(4);a=torch.randn(2,3,128,150);p=torch.randn(2,6,32,150);t=torch.zeros(2,3,150);t[:,0]=torch.arange(150)*.02
    for v in ('context_events','cross_events','band_events'):
        model=build(v);out=model.temporal(a,p,t);assert out.shape==(2,256) and torch.isfinite(out).all();out.square().mean().backward();assert any(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max()>0 for x in model.temporal.parameters());print('STRUCTURE_CHECK_PASS',v,flush=True)
    print('INVALID_RUN_ARCHIVED_AND_SEGMENT_IDENTITY_CHECKED',flush=True)
if __name__=='__main__':main()
