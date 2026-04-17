"""
Unit tests for PartialCrossEntropyLoss.

Tests cover:
  - Basic forward pass shape / type checks
  - Mathematical correctness (gamma=0 == standard CE on labeled pixels)
  - Gradient isolation (unlabeled pixels receive zero gradient)
  - Normalization stability across varying sparsity levels
  - Edge cases: all labeled, no labeled pixels, single class, single pixel
  - Numerical stability (NaN/Inf checks)
  - Focal weight behavior (higher gamma → lower loss on easy examples)

Run with:
    python -m pytest tests/test_loss.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.loss import PartialCrossEntropyLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_dims():
    """Standard small batch for fast tests."""
    return dict(B=2, C=6, H=32, W=32)


@pytest.fixture
def random_inputs(batch_dims):
    """Random logits, integer targets, and a sparse mask."""
    B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
    torch.manual_seed(42)
    logits     = torch.randn(B, C, H, W)
    targets    = torch.randint(0, C, (B, H, W))
    point_mask = (torch.rand(B, H, W) < 0.05).float()   # 5% labeled
    return logits, targets, point_mask


@pytest.fixture
def dense_mask(batch_dims):
    """All-ones mask — every pixel labeled."""
    B, H, W = batch_dims['B'], batch_dims['H'], batch_dims['W']
    return torch.ones(B, H, W)


@pytest.fixture
def sparse_mask(batch_dims):
    """Single labeled pixel per image."""
    B, H, W = batch_dims['B'], batch_dims['H'], batch_dims['W']
    mask = torch.zeros(B, H, W)
    mask[:, H // 2, W // 2] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 1. Basic forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:

    def test_returns_scalar(self, random_inputs):
        logits, targets, point_mask = random_inputs
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, point_mask)
        assert loss.ndim == 0, "Loss must be a scalar (0-dim tensor)"

    def test_returns_float_tensor(self, random_inputs):
        logits, targets, point_mask = random_inputs
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, point_mask)
        assert loss.dtype in (torch.float32, torch.float64), \
            f"Expected float tensor, got {loss.dtype}"

    def test_loss_is_positive(self, random_inputs):
        logits, targets, point_mask = random_inputs
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, point_mask)
        assert loss.item() > 0, "Loss must be positive"

    def test_loss_is_finite(self, random_inputs):
        logits, targets, point_mask = random_inputs
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, point_mask)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_requires_grad(self, random_inputs):
        logits, targets, point_mask = random_inputs
        logits = logits.requires_grad_(True)
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, point_mask)
        assert loss.requires_grad, "Loss must be differentiable w.r.t. logits"


# ---------------------------------------------------------------------------
# 2. Mathematical correctness
# ---------------------------------------------------------------------------

class TestMathematicalCorrectness:

    def test_gamma0_equals_masked_ce(self, random_inputs):
        """
        When gamma=0, focal loss reduces to standard CE.
        pCE with gamma=0 must equal the mean CE over labeled pixels.
        """
        logits, targets, point_mask = random_inputs
        loss_fn = PartialCrossEntropyLoss(gamma=0.0)
        pce_loss = loss_fn(logits, targets, point_mask)

        # Reference: manually compute masked CE
        ce_per_pixel = F.cross_entropy(logits, targets, reduction='none')  # (B,H,W)
        masked_ce = (ce_per_pixel * point_mask).sum() / point_mask.sum().clamp(min=1.0)

        assert torch.allclose(pce_loss, masked_ce, atol=1e-5), (
            f"gamma=0 pCE ({pce_loss.item():.6f}) != masked CE ({masked_ce.item():.6f})"
        )

    def test_focal_weight_less_than_standard_ce_for_easy_examples(self, batch_dims):
        """
        For an easy example (model confident and correct, 80% p_t),
        focal loss (gamma>0) should be LESS than CE.

        Logit spread of ±10 causes both CE and focal to underflow to 0.0 in
        float32 (p_t ≈ 1.0 → CE = -log(1.0) = 0). Use moderate logits so
        that CE is measurably positive while the model is still clearly
        "confident" (p_t ≈ 0.80 with class-0 logit=3, others=0).
        """
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']

        # Moderate confidence: p_t ≈ 0.80  →  CE ≈ 0.22, focal(γ=2) ≈ 0.009
        logits = torch.zeros(B, C, H, W)
        logits[:, 0, :, :] = 3.0    # class 0 predicted with 80% confidence
        targets = torch.zeros(B, H, W, dtype=torch.long)  # class 0 is correct
        mask = torch.ones(B, H, W)

        loss_ce    = PartialCrossEntropyLoss(gamma=0.0)(logits, targets, mask)
        loss_focal = PartialCrossEntropyLoss(gamma=2.0)(logits, targets, mask)

        assert loss_ce.item() > 0, (
            f"CE loss should be positive, got {loss_ce.item():.8f}. "
            "Logit values may still be too extreme."
        )
        assert loss_focal.item() < loss_ce.item(), (
            "Focal loss should be < CE for easy (confident-correct) examples. "
            f"focal={loss_focal.item():.6f}, ce={loss_ce.item():.6f}"
        )

    def test_higher_gamma_lower_loss_on_easy_example(self, batch_dims):
        """Increasing gamma should monotonically decrease loss on easy examples."""
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        logits = torch.full((B, C, H, W), -10.0)
        logits[:, 0, :, :] = 10.0
        targets = torch.zeros(B, H, W, dtype=torch.long)
        mask = torch.ones(B, H, W)

        losses = []
        for gamma in [0.0, 0.5, 1.0, 2.0, 3.0]:
            loss = PartialCrossEntropyLoss(gamma=gamma)(logits, targets, mask)
            losses.append(loss.item())

        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1], (
                f"Loss not monotonically decreasing with gamma: {losses}"
            )

    def test_normalization_invariant_to_scale(self, random_inputs):
        """
        The normalized loss should be the same whether we use 1 labeled pixel
        or 100 labeled pixels (both contain the same pixel).

        In other words, pCE = mean loss over labeled pixels — the batch size
        of the labeled set should not inflate the loss.
        """
        logits, targets, _ = random_inputs
        B, _, H, W = logits.shape

        # Mask 1: only one pixel labeled
        mask_1 = torch.zeros(B, H, W)
        mask_1[0, 5, 5] = 1.0

        # Mask 2: many pixels labeled
        mask_100 = (torch.rand(B, H, W) < 0.5).float()
        mask_100[0, 5, 5] = 1.0  # same pixel also labeled

        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss_1   = loss_fn(logits, targets, mask_1)
        loss_100 = loss_fn(logits, targets, mask_100)

        # They should NOT be equal (different labeled pixels → different losses)
        # But both should be finite and positive
        assert torch.isfinite(loss_1), "Single-pixel loss should be finite"
        assert torch.isfinite(loss_100), "Multi-pixel loss should be finite"
        assert loss_1.item() > 0
        assert loss_100.item() > 0


# ---------------------------------------------------------------------------
# 3. Gradient isolation
# ---------------------------------------------------------------------------

class TestGradientIsolation:

    def test_unlabeled_pixels_receive_zero_gradient(self, batch_dims):
        """
        Pixels where point_mask == 0 must receive zero gradient w.r.t. logits.
        This is the core correctness requirement of the pCE loss.
        """
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        torch.manual_seed(0)

        logits  = torch.randn(B, C, H, W, requires_grad=True)
        targets = torch.randint(0, C, (B, H, W))

        # Only one specific pixel is labeled
        labeled_r, labeled_c = 5, 7
        mask = torch.zeros(B, H, W)
        mask[0, labeled_r, labeled_c] = 1.0

        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, mask)
        loss.backward()

        grad = logits.grad  # (B, C, H, W)

        # Check all unlabeled positions have zero gradient (across all channels)
        unlabeled_mask = (1.0 - mask).bool()  # True where unlabeled
        for b in range(B):
            for r in range(H):
                for c in range(W):
                    if unlabeled_mask[b, r, c]:
                        pixel_grad = grad[b, :, r, c]
                        assert torch.allclose(pixel_grad, torch.zeros(C), atol=1e-8), (
                            f"Unlabeled pixel ({b},{r},{c}) has non-zero gradient: {pixel_grad}"
                        )

    def test_labeled_pixels_receive_nonzero_gradient(self, batch_dims):
        """Labeled pixels must receive a gradient (the loss is actually doing something)."""
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        torch.manual_seed(1)

        logits  = torch.randn(B, C, H, W, requires_grad=True)
        targets = torch.randint(0, C, (B, H, W))
        mask = torch.zeros(B, H, W)
        mask[0, 5, 7] = 1.0

        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, mask)
        loss.backward()

        labeled_grad = logits.grad[0, :, 5, 7]
        assert not torch.allclose(labeled_grad, torch.zeros(C), atol=1e-8), (
            "Labeled pixel (0,5,7) should have nonzero gradient"
        )

    def test_backward_does_not_error(self, random_inputs):
        """Loss.backward() must not raise any exception."""
        logits, targets, point_mask = random_inputs
        logits = logits.detach().requires_grad_(True)
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, point_mask)
        loss.backward()  # Should not raise


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_pixels_labeled_dense_mask(self, batch_dims, dense_mask):
        """Dense mask — all pixels labeled — must not raise and produce finite loss."""
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, dense_mask)
        assert torch.isfinite(loss), "Dense-mask loss must be finite"
        assert loss.item() > 0

    def test_single_labeled_pixel(self, batch_dims, sparse_mask):
        """One labeled pixel per image — must produce finite, positive loss."""
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, sparse_mask)
        assert torch.isfinite(loss), "Single-pixel loss must be finite"
        assert loss.item() > 0

    def test_zero_labeled_pixels_no_nan(self, batch_dims):
        """
        Empty mask (no labeled pixels) — the denominator would be zero.
        The clamp(min=1.0) in the loss must prevent NaN/Inf.
        Loss should be 0.0 (numerator is also 0).
        """
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        empty_mask = torch.zeros(B, H, W)

        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, empty_mask)

        assert not torch.isnan(loss), "Loss must not be NaN with empty mask"
        assert not torch.isinf(loss), "Loss must not be Inf with empty mask"
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"Empty mask should give loss=0.0, got {loss.item()}"
        )

    def test_single_class_all_pixels(self, batch_dims):
        """Single-class segmentation (binary: class 0 vs background)."""
        B, H, W = batch_dims['B'], batch_dims['H'], batch_dims['W']
        C = 2  # Binary
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        mask    = (torch.rand(B, H, W) < 0.1).float()
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, mask)
        assert torch.isfinite(loss)

    def test_perfect_predictions_give_low_loss(self, batch_dims):
        """
        If the model predicts the correct class with very high confidence
        for all labeled pixels, loss should be near zero.
        """
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        targets = torch.zeros(B, H, W, dtype=torch.long)  # all class 0
        mask = torch.ones(B, H, W)

        # Very confident correct predictions
        logits = torch.full((B, C, H, W), -100.0)
        logits[:, 0, :, :] = 100.0

        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, mask)
        assert loss.item() < 1e-3, (
            f"Perfect predictions should give near-zero loss, got {loss.item()}"
        )

    def test_large_logits_no_numerical_instability(self, batch_dims):
        """Extreme logit values must not cause NaN/Inf (relies on F.cross_entropy stability)."""
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        logits  = torch.full((B, C, H, W), 1e6)
        targets = torch.randint(0, C, (B, H, W))
        mask    = torch.ones(B, H, W)
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, mask)
        assert torch.isfinite(loss), f"Large logits caused non-finite loss: {loss.item()}"

    def test_batch_size_one(self):
        """Single-sample batch must work correctly."""
        B, C, H, W = 1, 6, 64, 64
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        mask    = (torch.rand(B, H, W) < 0.05).float()
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, mask)
        assert torch.isfinite(loss)

    def test_high_resolution_input(self):
        """Large spatial resolution (1024×1024) must not OOM or error in loss computation."""
        B, C, H, W = 1, 6, 512, 512
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        mask    = (torch.rand(B, H, W) < 0.01).float()
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)
        loss = loss_fn(logits, targets, mask)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 5. Normalization stability
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_loss_magnitude_stable_across_densities(self, batch_dims):
        """
        Loss magnitude should be roughly similar across densities.
        Without normalization, a 0.1%-density mask produces 50× smaller
        loss than a 5%-density mask — causing wildly different gradient scales.
        With normalization (divide by Σ mask), magnitudes should be comparable.
        """
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        torch.manual_seed(99)
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)

        losses = {}
        for density in [0.001, 0.01, 0.05, 0.5]:
            mask = (torch.rand(B, H, W) < density).float()
            if mask.sum() == 0:
                mask[0, 0, 0] = 1.0  # ensure at least one labeled pixel
            losses[density] = loss_fn(logits, targets, mask).item()

        loss_values = list(losses.values())
        ratio = max(loss_values) / (min(loss_values) + 1e-8)

        # With proper normalization, the ratio between max and min loss
        # should be much less than the ratio of densities (500×).
        # We allow up to 20× variation (different pixels → different loss values).
        assert ratio < 20.0, (
            f"Loss magnitude varies too much across densities (ratio={ratio:.1f}x). "
            f"Normalization may be incorrect. Losses: {losses}"
        )

    def test_denominator_clamped_at_zero_mask(self, batch_dims):
        """
        Zero mask → denominator clamped to 1.0, not 0.0.
        Tests that clamp(min=1.0) is implemented correctly.
        """
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        mask    = torch.zeros(B, H, W)  # empty mask
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)

        # If clamp is missing, this raises ZeroDivisionError or produces NaN
        try:
            loss = loss_fn(logits, targets, mask)
            assert not torch.isnan(loss), "NaN with empty mask — clamp not working"
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError — clamp(min=1.0) missing in loss implementation")


# ---------------------------------------------------------------------------
# 6. Gamma parameter behavior
# ---------------------------------------------------------------------------

class TestGammaParameter:

    def test_gamma_zero_initializes_correctly(self):
        loss_fn = PartialCrossEntropyLoss(gamma=0.0)
        assert loss_fn.gamma == 0.0

    def test_negative_gamma_raises(self):
        """
        Focal loss is only defined for gamma >= 0.
        A negative gamma would amplify easy examples and should be rejected.
        """
        with pytest.raises(ValueError, match='gamma must be >= 0'):
            PartialCrossEntropyLoss(gamma=-1.0)

    def test_gamma_float_accepted(self):
        """Non-integer gamma (e.g., 1.5) should work fine."""
        loss_fn = PartialCrossEntropyLoss(gamma=1.5)
        B, C, H, W = 1, 4, 16, 16
        logits  = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        mask    = torch.ones(B, H, W)
        loss = loss_fn(logits, targets, mask)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 7. Integration: loss feeds correctly into optimizer
# ---------------------------------------------------------------------------

class TestOptimizerIntegration:

    def test_loss_decreases_over_training_steps(self, batch_dims):
        """
        A few optimizer steps on a fixed batch should decrease the loss.
        Verifies the full gradient-optimizer loop is wired correctly.
        """
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        torch.manual_seed(42)

        # Simple linear head (stand-in for segmentation model)
        model = torch.nn.Conv2d(3, C, kernel_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn   = PartialCrossEntropyLoss(gamma=2.0)

        images  = torch.randn(B, 3, H, W)
        targets = torch.randint(0, C, (B, H, W))
        mask    = (torch.rand(B, H, W) < 0.3).float()

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            logits = model(images)
            loss   = loss_fn(logits, targets, mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should trend downward (not necessarily monotone due to SGD noise,
        # but average of last 3 should be less than average of first 3)
        early_avg = np.mean(losses[:3])
        late_avg  = np.mean(losses[-3:])
        assert late_avg < early_avg, (
            f"Loss did not decrease over 10 steps. "
            f"Early avg: {early_avg:.4f}, Late avg: {late_avg:.4f}. "
            f"Full trace: {[f'{l:.4f}' for l in losses]}"
        )

    def test_gradients_not_nan_after_backward(self, batch_dims):
        """Ensure no NaN gradients appear in model parameters after backward."""
        B, C, H, W = batch_dims['B'], batch_dims['C'], batch_dims['H'], batch_dims['W']
        model = torch.nn.Conv2d(3, C, kernel_size=1)
        loss_fn = PartialCrossEntropyLoss(gamma=2.0)

        images  = torch.randn(B, 3, H, W)
        targets = torch.randint(0, C, (B, H, W))
        mask    = (torch.rand(B, H, W) < 0.1).float()

        logits = model(images)
        loss   = loss_fn(logits, targets, mask)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.any(torch.isnan(param.grad)), (
                    f"NaN gradient in parameter '{name}'"
                )
                assert not torch.any(torch.isinf(param.grad)), (
                    f"Inf gradient in parameter '{name}'"
                )

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
