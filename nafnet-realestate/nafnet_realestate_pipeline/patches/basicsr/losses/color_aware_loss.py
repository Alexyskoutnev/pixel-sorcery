import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ColorAwareLoss(nn.Module):
    """Color-aware pixel loss for paired RGB enhancement.

    This is designed for cases where RGB+perceptual losses can still allow
    occasional chroma collapse or hue drift in high-chroma regions (e.g. painted
    walls becoming washed out).

    The loss is:
      L = loss_weight * (rgb_weight * L1(RGB) + cbcr_weight * L_cbcr_masked)

    Where L_cbcr_masked is an L1 loss on the Cb/Cr channels in YCbCr space,
    averaged only over pixels where the GT chroma magnitude exceeds a threshold.

    Notes:
    - Assumes inputs are RGB tensors in [0, 1].
    - Mask is computed from GT only (no gradient concerns).
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        rgb_weight: float = 1.0,
        cbcr_weight: float = 0.25,
        chroma_threshold: float = 0.06,
        mask_type: str = "hard",
        mask_strength: float = 25.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.rgb_weight = float(rgb_weight)
        self.cbcr_weight = float(cbcr_weight)
        self.chroma_threshold = float(chroma_threshold)
        self.mask_type = str(mask_type)
        self.mask_strength = float(mask_strength)
        self.eps = float(eps)

        if self.mask_type not in {"hard", "sigmoid"}:
            raise ValueError(f"mask_type must be 'hard' or 'sigmoid', got: {self.mask_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None, **kwargs) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target must have same shape, got {pred.shape} vs {target.shape}")
        if pred.dim() != 4 or pred.size(1) != 3:
            raise ValueError(f"expected (N, 3, H, W) RGB tensors, got {pred.shape}")

        loss_rgb = F.l1_loss(pred, target, reduction="mean")

        pred_ycbcr = rgb2ycbcr_pt(pred)
        target_ycbcr = rgb2ycbcr_pt(target)

        pred_cbcr = pred_ycbcr[:, 1:3, :, :]
        target_cbcr = target_ycbcr[:, 1:3, :, :]

        center = pred_cbcr.new_tensor(0.5)
        target_chroma_vec = target_cbcr - center
        target_chroma = torch.sqrt(target_chroma_vec.pow(2).sum(dim=1) + self.eps)  # (N, H, W)

        if self.mask_type == "hard":
            mask = (target_chroma > self.chroma_threshold).to(target_chroma.dtype)
        else:
            mask = torch.sigmoid(self.mask_strength * (target_chroma - self.chroma_threshold))

        cbcr_l1 = torch.abs(pred_cbcr - target_cbcr).mean(dim=1)  # (N, H, W)

        mask_sum = mask.sum()
        if mask_sum.item() > 0:
            loss_cbcr = (cbcr_l1 * mask).sum() / (mask_sum + self.eps)
        else:
            loss_cbcr = cbcr_l1.new_tensor(0.0)

        total = self.rgb_weight * loss_rgb + self.cbcr_weight * loss_cbcr
        return total * self.loss_weight

