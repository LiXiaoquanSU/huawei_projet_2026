"""Original preprocessing and submission interface, consolidated without math changes."""
import torch
from torch import nn

def standardize(x):
    x=x-x.mean(dim=-1,keepdim=True)
    return (x/torch.sqrt(x.square().mean(dim=-1,keepdim=True)+1e-5)).clamp(-6,6)

class TFPreprocessor(nn.Module):
    def forward(self,frames,intervals):
        b=frames.shape[0]
        p=frames[:,:,0,:,:,:].square().sum(-1)
        valid=(p>1e-8).to(frames.dtype)
        amp=.5*torch.log1p(p)
        amp=(amp*valid).reshape(b,600,3,128,8).sum(-1)/valid.reshape(b,600,3,128,8).sum(-1).clamp_min(1)
        image=amp.reshape(b,150,4,3,128).mean(2).permute(0,2,1,3)
        center=image.mean((2,3),keepdim=True)
        static=(image-center)/torch.sqrt((image-center).square().mean((2,3),keepdim=True)+1e-4)
        dynamic=image-image.mean(2,keepdim=True)
        dynamic=dynamic/torch.sqrt(dynamic.square().mean((2,3),keepdim=True)+1e-6)
        return torch.cat([static.clamp(-6,6),dynamic.clamp(-6,6)],dim=1)

class MixStyle(nn.Module):
    def forward(self,x):
        if not self.training:return x
        mean=x.mean((2,3),keepdim=True)
        std=torch.sqrt((x-mean).square().mean((2,3),keepdim=True)+1e-5)
        mean,std=mean.detach(),std.detach()
        order=torch.randperm(x.shape[0],device=x.device)
        weight=torch.distributions.Beta(.2,.2).sample((x.shape[0],1,1,1)).to(x.device)
        return (x-mean)/std*(weight*std+(1-weight)*std[order])+weight*mean+(1-weight)*mean[order]

class RelativePreprocessor(nn.Module):
    def forward(self,frames,intervals):
        b=frames.shape[0]
        r,i=frames[:,:,0,:,:,0],frames[:,:,0,:,:,1]
        power=r.square()+i.square()
        valid=(power>1e-8).to(frames.dtype)
        a=.5*torch.log1p(power)
        count=valid.reshape(b,600,3,32,32).sum(-1).clamp_min(1)
        amp=(a*valid).reshape(b,600,3,32,32).sum(-1)/count
        # Remove per-frame common gain across frequencies and per-bin static response.
        amp=amp-amp.mean(dim=-1,keepdim=True)
        amp=standardize(amp.reshape(b,600,96).transpose(1,2))
        r=r/torch.sqrt(power+1e-4);i=i/torch.sqrt(power+1e-4)
        phase=[]
        for u,v in [(0,1),(0,2),(1,2)]:
            mask=valid[:,:,u]*valid[:,:,v]
            denom=mask.reshape(b,600,32,32).sum(-1).clamp_min(1)
            c=((r[:,:,u]*r[:,:,v]+i[:,:,u]*i[:,:,v])*mask).reshape(b,600,32,32).sum(-1)/denom
            s=((i[:,:,u]*r[:,:,v]-r[:,:,u]*i[:,:,v])*mask).reshape(b,600,32,32).sum(-1)/denom
            mc,ms=c.mean(dim=1,keepdim=True),s.mean(dim=1,keepdim=True)
            norm=torch.sqrt(mc.square()+ms.square()+1e-4)
            aligned_c=(c*mc+s*ms)/norm
            aligned_s=(s*mc-c*ms)/norm
            phase.extend([aligned_c,aligned_s])
        phase=torch.stack(phase,dim=2).reshape(b,600,192).transpose(1,2)
        phase=standardize(phase)
        dt=intervals[:,:599].clamp_min(0)/1e6
        time=torch.stack([torch.log1p(dt/.005).clamp_max(8),(dt>.02).to(frames.dtype),(dt>0).to(frames.dtype)],dim=1)
        time=torch.cat([torch.zeros_like(time[:,:,:1]),time],dim=-1)
        return torch.cat([amp,phase,time],dim=1),torch.zeros_like(intervals[:,:17])

class CSIPreprocessor(nn.Module):
    """Batched input: [B,600,1,3,1024,2], [B,600]. No fitted statistics."""

    feature_channels = 195  # normalized log amplitude 96, derivative 96, timing 3
    summary_channels = 17

    def forward(self, frames, intervals):
        # Magnitude is invariant to a common complex phase rotation.
        real, imag = frames[:, :, 0, :, :, 0], frames[:, :, 0, :, :, 1]
        power = real.square() + imag.square()
        valid = (power > 1e-8).to(frames.dtype)
        log_amp = 0.5 * torch.log1p(power)
        batch = frames.shape[0]
        # Weighted averaging of 32 neighboring subcarriers into each of 32 bins.
        # Silent carriers are masked per input, never inferred from validation data.
        sums = (log_amp * valid).reshape(batch, 600, 3, 32, 32).sum(dim=-1)
        weights = valid.reshape(batch, 600, 3, 32, 32).sum(dim=-1)
        pooled = sums / weights.clamp_min(1.0)
        active = (weights > 0).to(frames.dtype)
        count = active.sum(dim=(1, 3)).clamp_min(1.0)  # B,Rx
        mean = (pooled * active).sum(dim=(1, 3)) / count
        variance = ((pooled - mean[:, None, :, None]).square() * active).sum(dim=(1, 3)) / count
        # Temporal centering removes the static frequency response from this branch.
        center = (pooled * active).sum(dim=1, keepdim=True) / active.sum(dim=1, keepdim=True).clamp_min(1)
        centered = (pooled - center) * active
        motion = torch.sqrt(centered.square().sum(dim=(1, 3)) / count + 1e-8)
        normalized = (centered / motion[:, None, :, None].clamp_min(0.02)).clamp(-8, 8)

        # The last interval is padding and is deliberately never read.
        dt = intervals[:, :599].clamp_min(0) / 1e6
        valid_dt = dt > 0
        # Normalize differences to a nominal 5ms step; cap gap effect and extremes.
        scale = 0.005 / dt.clamp(0.001, 0.1)
        derivative = ((pooled[:, 1:] - pooled[:, :-1]) * scale[:, :, None, None]).clamp(-2, 2)
        pair_mask = active[:, 1:] * active[:, :-1] * valid_dt[:, :, None, None]
        derivative = derivative * pair_mask
        derivative = torch.cat((torch.zeros_like(pooled[:, :1]), derivative), dim=1)
        valid_count = valid_dt.to(frames.dtype).sum(dim=1).clamp_min(1)
        dt_mean = dt.sum(dim=1) / valid_count
        dt_std = torch.sqrt(((dt - dt_mean[:, None]).square() * valid_dt).sum(dim=1) / valid_count + 1e-12)
        # Aligned with the receiving frame; first frame has no preceding interval.
        timing = torch.stack((torch.log1p(dt / 0.005).clamp_max(8),
                              (dt > 0.02).to(frames.dtype), valid_dt.to(frames.dtype)), dim=-1)
        timing = torch.cat((torch.zeros_like(timing[:, :1]), timing), dim=1)
        temporal = torch.cat((normalized.reshape(batch, 600, 96),
                              (derivative / 0.02).clamp(-8, 8).reshape(batch, 600, 96), timing), dim=-1).transpose(1, 2)
        derivative_strength = derivative.abs().mean(dim=(1, 3))
        summary = torch.cat((mean / 8, torch.sqrt(variance + 1e-8),
                             torch.log1p(motion / 0.01),
                             torch.log1p(derivative_strength / 0.005),
                             torch.log1p(dt_mean[:, None] / 0.005),
                             (dt_std / dt_mean.clamp_min(1e-6)).clamp_max(20)[:, None],
                             torch.log1p(dt.max(dim=1).values[:, None] / 0.005),
                             (dt > 0.02).to(frames.dtype).mean(dim=1, keepdim=True),
                             torch.log1p(dt.sum(dim=1, keepdim=True))), dim=1)
        return temporal, summary

class SubmissionModel(nn.Module):
    """Official unbatched signature: frames, intervals -> float32 logits (5,)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames, intervals):
        return self.model(frames.unsqueeze(0), intervals.unsqueeze(0)).squeeze(0)

def export_submission(model, example, path):
    """Export CPU eval model with numerical stabilizers preserved."""
    submission = SubmissionModel(model.cpu().eval()).eval()
    torch.onnx.export(submission, tuple(x.cpu() for x in example), str(path),
                      input_names=['frames', 'intervals'], output_names=['logits'],
                      opset_version=18, external_data=False, optimize=False)

def combine(tf,phase):
    b=tf.shape[0]
    amplitude=tf.reshape(b,6,150,32,4).mean(-1).permute(0,1,3,2)
    relative=phase[:,96:288].reshape(b,6,32,150,4).mean(-1)
    return torch.cat([amplitude,relative],1).reshape(b,384,150)
