"""
tests/test_residual_knowledge.py
───────────────────────────────────────────────
Unit tests for the Residual Knowledge (RK) metric.
Validates correctness under identical models, extreme cases,
unlearned accuracy lower than retrained, reproducibility,
and dimension compatibility.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.residual_knowledge import compute_residual_knowledge

# ─────────────────────────────────────────────
# Mock Models for testing
# ─────────────────────────────────────────────

class AlwaysCorrectModel(nn.Module):
    """Always predicts the true class (assumed to be class 1)."""
    def forward(self, x):
        # Output shape: [batch_size, 3]
        out = torch.zeros(x.size(0), 3, device=x.device)
        out[:, 1] = 10.0  # Logits for class 1 are high
        return out

class AlwaysWrongModel(nn.Module):
    """Always predicts class 0 (incorrect if true class is 1)."""
    def forward(self, x):
        out = torch.zeros(x.size(0), 3, device=x.device)
        out[:, 0] = 10.0  # Logits for class 0 are high
        return out

class SignBasedModel(nn.Module):
    """Predicts class 1 if the first feature is positive, else class 0."""
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        pred = (x_flat[:, 0] > 0.0).long()
        out = torch.zeros(x.size(0), 3, device=x.device)
        out[torch.arange(x.size(0)), pred] = 10.0
        return out

# ─────────────────────────────────────────────
# Unit Tests
# ─────────────────────────────────────────────

def test_identical_models():
    """
    Test 1: Identical models (model_unlearned == model_retrained)
    should return RK = 1.0 for samples where r_i > 0.
    """
    # Create simple dataset of 5 samples, all with label 1 (correct class)
    X = torch.zeros(5, 2)
    y = torch.ones(5, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)

    model = AlwaysCorrectModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = compute_residual_knowledge(
        model_unlearned=model,
        model_retrained=model,
        forget_loader=loader,
        tau=0.03,
        c=50,
        device=device,
        seed=42
    )

    # For AlwaysCorrectModel, r_i should be 50 (all correct)
    assert all(r == 50 for r in res["r_counts"])
    assert all(u == 50 for u in res["u_counts"])
    assert all(val == 1.0 for val in res["rk_tau_per_sample"])
    assert res["rk_tau_forget_set"] == 1.0


def test_extreme_case_undefined_ratio():
    """
    Test 2: Extreme case where r_i = 0 and u_i = c.
    The ratio should be NaN for that sample, and propagate to macro average.
    """
    X = torch.zeros(3, 2)
    y = torch.ones(3, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=3)

    model_unlearned = AlwaysCorrectModel()
    model_retrained = AlwaysWrongModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = compute_residual_knowledge(
        model_unlearned=model_unlearned,
        model_retrained=model_retrained,
        forget_loader=loader,
        tau=0.05,
        c=10,
        device=device,
        seed=123
    )

    # r_i should be 0, u_i should be c (10)
    assert all(r == 0 for r in res["r_counts"])
    assert all(u == 10 for u in res["u_counts"])
    
    # Each sample's RK must be NaN (as r_i = 0)
    import math
    assert all(math.isnan(val) for val in res["rk_tau_per_sample"])
    assert math.isnan(res["rk_tau_forget_set"])


def test_extreme_case_zero_zero():
    """
    Test 3: Extreme case where r_i = 0 and u_i = 0.
    The ratio should be NaN for that sample, and propagate to macro average.
    """
    X = torch.zeros(2, 2)
    y = torch.ones(2, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)

    model_unlearned = AlwaysWrongModel()
    model_retrained = AlwaysWrongModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = compute_residual_knowledge(
        model_unlearned=model_unlearned,
        model_retrained=model_retrained,
        forget_loader=loader,
        tau=0.1,
        c=20,
        device=device,
        seed=456
    )

    # Both r_i and u_i should be 0
    assert all(r == 0 for r in res["r_counts"])
    assert all(u == 0 for u in res["u_counts"])
    
    import math
    assert all(math.isnan(val) for val in res["rk_tau_per_sample"])
    assert math.isnan(res["rk_tau_forget_set"])


def test_unlearned_performs_worse():
    """
    Test 4: Unlearned model performs worse than retrained (u_i < r_i).
    The ratio RK should be < 1.0.
    """
    # X is 0. With Gaussian noise around 0, the first feature will be positive 
    # approximately 50% of the time, so SignBasedModel will be correct ~50% of the time.
    X = torch.zeros(4, 2)
    y = torch.ones(4, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)

    # model_unlearned is correct ~50% of the time, model_retrained is correct 100% of the time
    model_unlearned = SignBasedModel()
    model_retrained = AlwaysCorrectModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = compute_residual_knowledge(
        model_unlearned=model_unlearned,
        model_retrained=model_retrained,
        forget_loader=loader,
        tau=0.03,
        c=100,
        device=device,
        seed=0
    )

    # Check that retrained is always correct
    assert all(r == 100 for r in res["r_counts"])
    # Check that unlearned is correct only sometimes (u_i < 100)
    assert all(u < 100 for u in res["u_counts"])
    assert all(u > 0 for u in res["u_counts"]) # check it's not zero to avoid NaN

    # Check RK is less than 1.0
    assert all(val < 1.0 for val in res["rk_tau_per_sample"])
    assert res["rk_tau_forget_set"] < 1.0


def test_reproducibility():
    """
    Test 5: Verify reproducibility under the same seed,
    and difference under a different seed.
    """
    X = torch.zeros(5, 2)
    y = torch.ones(5, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    model_unlearned = SignBasedModel()
    model_retrained = AlwaysCorrectModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run twice with the same seed
    res1 = compute_residual_knowledge(
        model_unlearned=model_unlearned,
        model_retrained=model_retrained,
        forget_loader=loader,
        tau=0.03,
        c=50,
        device=device,
        seed=777
    )

    res2 = compute_residual_knowledge(
        model_unlearned=model_unlearned,
        model_retrained=model_retrained,
        forget_loader=loader,
        tau=0.03,
        c=50,
        device=device,
        seed=777
    )

    # Run once with a different seed
    res3 = compute_residual_knowledge(
        model_unlearned=model_unlearned,
        model_retrained=model_retrained,
        forget_loader=loader,
        tau=0.03,
        c=50,
        device=device,
        seed=888
    )

    # Identical seed -> identical results
    assert res1["rk_tau_forget_set"] == res2["rk_tau_forget_set"]
    assert res1["u_counts"] == res2["u_counts"]
    assert res1["r_counts"] == res2["r_counts"]

    # Different seed -> should yield different counts/results (with high probability)
    assert res1["u_counts"] != res3["u_counts"]


def test_dimension_compatibility():
    """
    Test dimension compatibility with 2D data (spiral) and 4D data (cifar_nano).
    Also ensures image inputs are clamped to [0, 1] while 2D coordinates are not.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 2D Coordinates (Spiral shape: [B, 2])
    # Values should not be clamped (can be outside [0, 1])
    X_2d = torch.full((2, 2), 5.0)  # Starting at 5.0
    y_2d = torch.ones(2, dtype=torch.long)
    loader_2d = DataLoader(TensorDataset(X_2d, y_2d), batch_size=2)

    # We want to check if the perturbations ever go below or above [0, 1] range.
    # To check this, let's capture the inputs in a mock model.
    perturbed_inputs = []
    class CaptureModel(nn.Module):
        def forward(self, x):
            perturbed_inputs.append(x.cpu().clone())
            return torch.zeros(x.size(0), 3, device=x.device)

    model_capture = CaptureModel()
    compute_residual_knowledge(
        model_unlearned=model_capture,
        model_retrained=model_capture,
        forget_loader=loader_2d,
        tau=2.0,  # High tau to ensure large noise deviation
        c=10,
        device=device,
        seed=42
    )

    # Check that coordinate perturbations were NOT clamped (should remain around 5.0)
    for px in perturbed_inputs:
        assert (px > 1.0).any()  # Definitely not clamped to [0, 1]

    # 2. 4D Images (cifar_nano shape: [B, 3, 8, 8])
    # Values MUST be clamped to [0, 1]
    perturbed_inputs.clear()
    X_4d = torch.full((2, 3, 8, 8), 0.9)  # Starting near 1.0
    y_4d = torch.ones(2, dtype=torch.long)
    loader_4d = DataLoader(TensorDataset(X_4d, y_4d), batch_size=2)

    compute_residual_knowledge(
        model_unlearned=model_capture,
        model_retrained=model_capture,
        forget_loader=loader_4d,
        tau=2.0,  # High noise to force clamping
        c=10,
        device=device,
        seed=42
    )

    # Check that image perturbations WERE clamped to [0.0, 1.0]
    for px in perturbed_inputs:
        assert (px >= 0.0).all()
        assert (px <= 1.0).all()
