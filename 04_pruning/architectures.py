"""Alternative segmentation heads for the YOLO11s multitask model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionHead(nn.Module):
    def __init__(self, feature_indices: tuple[int, ...], input_channels: tuple[int, ...],
                 width: int, upsample_output: bool = False):
        super().__init__()
        if len(feature_indices) != len(input_channels) or len(feature_indices) < 2:
            raise ValueError("Feature indices and input channels must match")
        self.feature_indices = feature_indices
        self.upsample_output = upsample_output
        self.lateral = nn.ModuleList(nn.Conv2d(channels, width, 1) for channels in input_channels)
        self.refine = nn.ModuleList(nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.SiLU(inplace=True))
            for _ in range(len(feature_indices) - 1))
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, features: dict[int, torch.Tensor]) -> torch.Tensor:
        skips = [layer(features[index]) for layer, index in zip(self.lateral, self.feature_indices)]
        x = skips[-1]
        for skip, block in zip(reversed(skips[:-1]), self.refine):
            x = block(F.interpolate(x, size=skip.shape[-2:], mode="nearest") + skip)
        logits = self.out(x)
        if self.upsample_output:
            logits = F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)
        return logits


@torch.no_grad()
def transfer_width64(source, target: FeatureFusionHead) -> list[int]:
    if len(target.lateral) != 4 or target.out.in_channels != 64:
        raise ValueError("Width transfer requires a four-scale, 64-channel head")
    scores = source.out.weight[0, :, 0, 0].abs().clone()
    for block in source.refine:
        scores += block[1].weight.abs()
    indices = torch.topk(scores, 64).indices.sort().values
    for old, new in zip(source.lateral, target.lateral):
        new.weight.copy_(old.weight.index_select(0, indices))
        new.bias.copy_(old.bias.index_select(0, indices))
    for old, new in zip(source.refine, target.refine):
        new[0].weight.copy_(old[0].weight.index_select(0, indices).index_select(1, indices))
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(new[1], name).copy_(getattr(old[1], name).index_select(0, indices))
        new[1].num_batches_tracked.copy_(old[1].num_batches_tracked)
    target.out.weight.copy_(source.out.weight.index_select(1, indices))
    target.out.bias.copy_(source.out.bias)
    return indices.cpu().tolist()


@torch.no_grad()
def transfer_p3(source, target: FeatureFusionHead):
    if target.feature_indices != (16, 19, 22) or target.out.in_channels != 128:
        raise ValueError("P3 transfer requires the original 128-channel width")
    for old, new in zip(source.lateral[1:], target.lateral):
        new.load_state_dict(old.state_dict())
    for old, new in zip(source.refine[:2], target.refine):
        new.load_state_dict(old.state_dict())
    target.out.load_state_dict(source.out.state_dict())
