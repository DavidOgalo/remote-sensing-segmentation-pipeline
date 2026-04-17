"""
PartialCrossEntropyLoss — standalone importable module.

Formula:
    pCE = Σ(FocalLoss(pred, GT) × MASK_labeled) / Σ(MASK_labeled)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross-Entropy Loss for weakly-supervised semantic segmentation.

    Enables training on sparse point annotations by computing loss only
    at labeled pixel locations. Unlabeled pixels contribute zero gradient,
    leaving their predictions free to be shaped by labeled neighbors via
    the network's spatial inductive bias (receptive field of the encoder).

    When gamma=0, this reduces exactly to masked standard cross-entropy.
    When gamma>0, focal weighting down-weights easy (confident-correct)
    labeled pixels so the sparse gradient budget focuses on hard examples.

    Args:
        gamma (float):       Focal loss focusing parameter (>= 0). Default: 2.0
        ignore_index (int):  Class index to exclude from loss (e.g. void/border).
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        if gamma < 0:
            raise ValueError(f'gamma must be >= 0. Got: {gamma}')
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def focal_loss_per_pixel(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute element-wise focal loss without reduction.

        Args:
            logits:  (B, C, H, W) raw model outputs (pre-softmax)
            targets: (B, H, W)    integer ground-truth class indices
        Returns:
            (B, H, W) float tensor of per-pixel focal loss values
        """
        # Per-pixel CE = -log(p_t), shape (B, H, W)
        ce = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        if self.gamma == 0.0:
            return ce

        # p_t = softmax probability of the true class = exp(-CE)
        pt = torch.exp(-ce)

        # Focal weight: (1 - p_t)^gamma
        #   - p_t → 1 (easy, confident correct) → weight → 0
        #   - p_t → 0 (hard, wrong prediction)  → weight → 1
        focal_weight = (1.0 - pt) ** self.gamma

        return focal_weight * ce

    def forward(
        self,
        logits:     torch.Tensor,
        targets:    torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute partial cross-entropy loss over labeled pixels only.

        Args:
            logits:     (B, C, H, W) model predictions (raw, pre-softmax)
            targets:    (B, H, W)    integer ground-truth labels [0, C-1]
            point_mask: (B, H, W)    float32 binary; 1=labeled, 0=unlabeled
        Returns:
            Scalar loss value (differentiable w.r.t. logits)
        """
        # Per-pixel focal loss — (B, H, W)
        pixel_loss = self.focal_loss_per_pixel(logits, targets)

        # Zero out gradient at unlabeled pixels
        masked_loss = pixel_loss * point_mask

        # Normalize by labeled pixel count — prevents magnitude from
        # depending on annotation density (critical for stable training)
        n_labeled = point_mask.sum().clamp(min=1.0)
        loss      = masked_loss.sum() / n_labeled

        return loss

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'gamma={self.gamma}, ignore_index={self.ignore_index})'
        )
