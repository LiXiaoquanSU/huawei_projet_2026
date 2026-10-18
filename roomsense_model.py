"""Small amplitude/ATI baseline. All inference preprocessing is inside ONNX."""

import torch
from torch import nn


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


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation, groups=width),
            nn.Conv1d(width, width, 1), nn.GroupNorm(8, width), nn.SiLU(), nn.Dropout(0.1))

    def forward(self, x):
        return x + self.block(x)


class RoomSenseModel(nn.Module):
    """Train with a batch dimension; returns [B,5] logits for headcounts 0..4."""

    def __init__(self, width=64):
        super().__init__()
        self.preprocess = CSIPreprocessor()
        self.stem = nn.Sequential(nn.Conv1d(195, width, 5, stride=2, padding=2),
                                  nn.GroupNorm(8, width), nn.SiLU())
        self.temporal = nn.Sequential(*(TemporalBlock(width, d) for d in [1, 2, 4, 8, 16]))
        self.summary = nn.Sequential(nn.Linear(17, 32), nn.LayerNorm(32), nn.SiLU())
        self.classifier = nn.Sequential(nn.Linear(width*3 + 32, 64), nn.SiLU(),
                                        nn.Dropout(0.2), nn.Linear(64, 5))

    def forward_features(self, temporal, summary):
        x = self.temporal(self.stem(temporal))
        mean = x.mean(dim=-1)
        std = torch.sqrt((x - mean[:, :, None]).square().mean(dim=-1) + 1e-5)
        aggregate = torch.cat((mean, std, x.max(dim=-1).values, self.summary(summary)), dim=1)
        return self.classifier(aggregate)

    def forward(self, frames, intervals):
        return self.forward_features(*self.preprocess(frames, intervals))


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
