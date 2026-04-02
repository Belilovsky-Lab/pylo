"""Tests for CELO2 optimizer implementation."""

import copy

import torch
import torch.nn as nn
import pytest

from pylo.optim.Celo2_naive import Celo2_naive, _is_adamw_param


class SimpleMLP(nn.Module):
    def __init__(self, d_in=20, d_hidden=64, d_out=10):
        super().__init__()
        self.input_layer = nn.Linear(d_in, d_hidden)
        self.hidden1 = nn.Linear(d_hidden, d_hidden)
        self.hidden2 = nn.Linear(d_hidden, d_hidden)
        self.output_layer = nn.Linear(d_hidden, d_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        return self.output_layer(x)


# ── Parameter grouping tests ──────────────────────────────────────────────────


class TestParamGrouping:
    def test_1d_params_get_adamw(self):
        model = SimpleMLP()
        opt = Celo2_naive(model.parameters(), lr=1e-3)
        opt.set_param_names(model)

        for name, p in model.named_parameters():
            is_adamw = opt._param_is_adamw[id(p)]
            if p.ndim <= 1:
                assert is_adamw, f"{name} (1D) should use AdamW"

    def test_input_output_layers_get_adamw(self):
        model = SimpleMLP()
        opt = Celo2_naive(model.parameters(), lr=1e-3)
        opt.set_param_names(model)

        param_list = list(model.parameters())
        # First 2D param (input_layer.weight) should be AdamW
        assert opt._param_is_adamw[id(param_list[0])], "Input layer weight should use AdamW"
        # Last 2D param (output_layer.weight) — it's the second-to-last param (last is bias)
        # Actually the last param in the list is output_layer.bias (1D), the one before is output_layer.weight (2D)
        output_weight = dict(model.named_parameters())["output_layer.weight"]
        assert opt._param_is_adamw[id(output_weight)], "Output layer weight should use AdamW"

    def test_hidden_layers_get_celo2(self):
        model = SimpleMLP()
        opt = Celo2_naive(model.parameters(), lr=1e-3)
        opt.set_param_names(model)

        for name in ["hidden1.weight", "hidden2.weight"]:
            p = dict(model.named_parameters())[name]
            assert not opt._param_is_adamw[id(p)], f"{name} should use CELO2"

    def test_embedding_gets_adamw(self):
        """Params with 'embed' in name should use AdamW."""
        assert _is_adamw_param("token_embedding.weight", 1, 10, torch.randn(100, 64))


# ── Convergence tests ─────────────────────────────────────────────────────────


class TestConvergenceLinearRegression:
    """Train a linear model on synthetic regression data. Should converge."""

    def test_linear_regression_converges(self):
        torch.manual_seed(42)
        d_in, d_out = 10, 5
        n_samples = 200
        n_steps = 300

        # Generate data
        w_true = torch.randn(d_in, d_out) * 0.5
        b_true = torch.randn(d_out) * 0.1
        X = torch.randn(n_samples, d_in)
        Y = X @ w_true + b_true + torch.randn(n_samples, d_out) * 0.01

        # Model — single hidden layer so there's a CELO2 param
        model = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, d_out),
        )
        optimizer = Celo2_naive(model.parameters(), lr=1e-3)
        optimizer.set_param_names(model)
        loss_fn = nn.MSELoss()

        losses = []
        for step in range(n_steps):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Should converge: final loss much less than initial
        assert losses[-1] < losses[0] * 0.1, (
            f"Loss did not converge: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )
        # No NaN/Inf
        assert all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in losses), (
            "NaN or Inf detected in losses"
        )

    def test_no_nan_in_parameters(self):
        torch.manual_seed(123)
        model = SimpleMLP(d_in=10, d_hidden=32, d_out=5)
        optimizer = Celo2_naive(model.parameters(), lr=1e-3)
        optimizer.set_param_names(model)

        X = torch.randn(50, 10)
        Y = torch.randn(50, 5)

        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X), Y)
            loss.backward()
            optimizer.step()

        for name, p in model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in {name}"
            assert not torch.isinf(p).any(), f"Inf in {name}"


class TestConvergenceMLP:
    """Train a small MLP on synthetic classification. Should improve from random."""

    def test_mlp_classification_converges(self):
        torch.manual_seed(42)
        d_in = 20
        d_out = 5
        n_samples = 500
        n_steps = 500

        # Synthetic classification: clustered data
        X = torch.randn(n_samples, d_in)
        # Create labels from a simple linear classifier
        w_true = torch.randn(d_in, d_out)
        logits_true = X @ w_true
        Y = logits_true.argmax(dim=1)

        model = SimpleMLP(d_in=d_in, d_hidden=64, d_out=d_out)
        optimizer = Celo2_naive(model.parameters(), lr=1e-3)
        optimizer.set_param_names(model)
        loss_fn = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None

        for step in range(n_steps):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            if step == n_steps - 1:
                final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.3, (
            f"Classification loss did not converge: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

        # Check accuracy improved from random (~20% for 5 classes)
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
            accuracy = (preds == Y).float().mean().item()

        assert accuracy > 0.5, f"Accuracy too low: {accuracy:.2%} (expected >50%)"


# ── State save/load tests ─────────────────────────────────────────────────────


class TestStateSaveLoad:
    def test_optimizer_state_roundtrip(self):
        torch.manual_seed(42)
        model = SimpleMLP(d_in=10, d_hidden=32, d_out=5)
        optimizer = Celo2_naive(model.parameters(), lr=1e-3)
        optimizer.set_param_names(model)

        X = torch.randn(32, 10)
        Y = torch.randn(32, 5)

        # Train for 5 steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X), Y)
            loss.backward()
            optimizer.step()

        # Save state (deepcopy to avoid mutation by continued training)
        opt_state = copy.deepcopy(optimizer.state_dict())
        model_state = copy.deepcopy(model.state_dict())

        # Train for 5 more steps and record losses
        reference_losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X), Y)
            loss.backward()
            optimizer.step()
            reference_losses.append(loss.item())

        # Reload from checkpoint
        model2 = SimpleMLP(d_in=10, d_hidden=32, d_out=5)
        model2.load_state_dict(model_state)
        optimizer2 = Celo2_naive(model2.parameters(), lr=1e-3)
        optimizer2.set_param_names(model2)
        optimizer2.load_state_dict(opt_state)

        # Train for 5 more steps — should match reference
        resumed_losses = []
        for _ in range(5):
            optimizer2.zero_grad()
            loss = nn.MSELoss()(model2(X), Y)
            loss.backward()
            optimizer2.step()
            resumed_losses.append(loss.item())

        for i, (ref, res) in enumerate(zip(reference_losses, resumed_losses)):
            assert abs(ref - res) < 1e-5, (
                f"Step {i}: reference loss {ref:.6f} != resumed loss {res:.6f}"
            )


# ── Dense vs per-group equivalence test ──────────────────────────────────────


class TestDenseEquivalence:
    """Verify that the single dense first layer produces identical outputs
    to the old per-group weight approach."""

    def test_dense_matches_per_group(self):
        torch.manual_seed(0)
        from pylo.models.Celo2_MLP import Celo2MLP, DEFAULT_INPUT_GROUP_DIMS

        hidden_size = 8
        input_group_dims = DEFAULT_INPUT_GROUP_DIMS  # [1,1,1,3,1,3,1,3,1,3,3,3,3,3]
        total_dim = sum(input_group_dims)  # 30

        # Build the new dense model with random weights
        model = Celo2MLP()
        nn.init.normal_(model.first_layer.weight)
        nn.init.normal_(model.first_layer.bias)
        nn.init.normal_(model.hidden.weight)
        nn.init.normal_(model.hidden.bias)
        nn.init.normal_(model.output.weight)
        nn.init.normal_(model.output.bias)

        # Extract per-group weight slices from the dense weight matrix
        # dense weight shape: (hidden_size, 30)
        dense_w = model.first_layer.weight.data.clone()
        dense_b = model.first_layer.bias.data.clone()

        # Manually compute the per-group forward pass (old approach)
        batch = 16
        input_groups = []
        offset = 0
        for dim in input_group_dims:
            input_groups.append(torch.randn(batch, dim))
            offset += dim

        # Old approach: sum of per-group matmuls
        old_out = torch.zeros(batch, hidden_size)
        col = 0
        for inp, dim in zip(input_groups, input_group_dims):
            w_group = dense_w[:, col : col + dim]  # (hidden_size, dim)
            old_out = old_out + inp @ w_group.t()
            col += dim
        old_out = torch.relu(old_out + dense_b)

        # New approach: concat then single matmul
        x_cat = torch.cat(input_groups, dim=-1)  # (batch, 30)
        new_out = torch.relu(x_cat @ dense_w.t() + dense_b)

        assert torch.allclose(old_out, new_out, atol=1e-6), (
            f"Max diff: {(old_out - new_out).abs().max().item()}"
        )

        # Also verify through the full model forward pass
        model.eval()
        with torch.no_grad():
            full_out = model(input_groups)

        # Manually compute full forward with old-style first layer
        with torch.no_grad():
            after_hidden = torch.relu(model.hidden(old_out))
            expected_out = model.output(after_hidden)

        assert torch.allclose(full_out, expected_out, atol=1e-6), (
            f"Full model max diff: {(full_out - expected_out).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
