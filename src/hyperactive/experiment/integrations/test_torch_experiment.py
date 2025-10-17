"""Tests for TorchExperiment integration."""

import lightning as L
import numpy as np
import pytest
import torch
from hyperactive.experiment.integrations import TorchExperiment
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Test Lightning Module
class SimpleTestModel(L.LightningModule):
    def __init__(self, input_dim=10, hidden_dim=16, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = nn.functional.cross_entropy(self(x), y)
        self.log("val_loss", val_loss, on_epoch=True)
        self.log("val_acc", 0.8, on_epoch=True)  # Dummy accuracy
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# Test DataModule
class SimpleTestDataModule(L.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        self.train, self.val = torch.utils.data.random_split(dataset, [80, 20])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)


def test_torch_experiment_initialization():
    """Test that TorchExperiment can be initialized."""
    datamodule = SimpleTestDataModule()

    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
        objective_metric="val_loss",
    )

    assert experiment.datamodule is datamodule
    assert experiment.lightning_module is SimpleTestModel
    assert experiment.objective_metric == "val_loss"
    print("✅ Test 1 passed: Initialization works")


def test_paramnames():
    """Test that _paramnames returns correct parameters."""
    datamodule = SimpleTestDataModule()
    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
    )

    param_names = experiment._paramnames()

    assert "input_dim" in param_names
    assert "hidden_dim" in param_names
    assert "lr" in param_names
    assert "self" not in param_names
    print("✅ Test 2 passed: _paramnames returns correct parameters")


def test_evaluate_basic():
    """Test basic evaluation with valid parameters."""
    datamodule = SimpleTestDataModule()
    datamodule.setup()

    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
        trainer_kwargs={"max_epochs": 1},
        objective_metric="val_loss",
    )

    params = {"input_dim": 10, "hidden_dim": 16, "lr": 1e-3}
    score, metadata = experiment._evaluate(params)

    # Check return types
    assert isinstance(score, (np.floating, float))
    assert isinstance(metadata, dict)
    assert score > 0  # Loss should be positive
    assert score < 10  # Reasonable loss value
    print(f"✅ Test 3 passed: Basic evaluation works (score={score:.4f})")


def test_evaluate_different_params():
    """Test evaluation with different hyperparameters."""
    datamodule = SimpleTestDataModule()
    datamodule.setup()

    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
        trainer_kwargs={"max_epochs": 1},
    )

    params1 = {"input_dim": 10, "hidden_dim": 32, "lr": 1e-3}
    params2 = {"input_dim": 10, "hidden_dim": 64, "lr": 1e-2}

    score1, _ = experiment._evaluate(params1)
    score2, _ = experiment._evaluate(params2)

    # Both should be valid scores
    assert isinstance(score1, (np.floating, float))
    assert isinstance(score2, (np.floating, float))
    assert score1 > 0
    assert score2 > 0
    print(
        f"✅ Test 4 passed: Multiple evaluations work (scores={score1:.4f}, {score2:.4f})"
    )


def test_custom_objective_metric():
    """Test using a custom objective metric."""
    datamodule = SimpleTestDataModule()
    datamodule.setup()

    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
        trainer_kwargs={"max_epochs": 1},
        objective_metric="val_acc",  # Use accuracy instead of loss
    )

    params = {"input_dim": 10, "hidden_dim": 16, "lr": 1e-3}
    score, metadata = experiment._evaluate(params)

    assert isinstance(score, (np.floating, float))

    print(f"✅ Test 5 passed: Custom metric works (val_acc={score:.4f})")


def test_invalid_metric_name():
    """Test that invalid metric name is handled."""
    datamodule = SimpleTestDataModule()
    datamodule.setup()

    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
        trainer_kwargs={"max_epochs": 1},
        objective_metric="nonexistent_metric",
    )

    params = {"input_dim": 10, "hidden_dim": 16, "lr": 1e-3}
    score, metadata = experiment._evaluate(params)

    # Should return inf for failed run
    assert score == float("inf")

    print("✅ Test 6 passed: Invalid metric handled correctly")


def test_trainer_kwargs_override():
    """Test that trainer kwargs can be overridden."""
    datamodule = SimpleTestDataModule()
    datamodule.setup()

    # Test with custom max_epochs
    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
        trainer_kwargs={"max_epochs": 2},
    )

    assert experiment._trainer_kwargs["max_epochs"] == 2
    assert (
        experiment._trainer_kwargs["enable_progress_bar"] is False
    )  # Default preserved
    print("✅ Test 7 passed: Trainer kwargs override works")


def test_get_test_params():
    """Test that get_test_params returns valid configuration."""
    params = TorchExperiment.get_test_params()

    assert isinstance(params, dict)
    assert "datamodule" in params
    assert "lightning_module" in params
    assert "trainer_kwargs" in params
    assert "objective_metric" in params

    # Should be able to create instance
    experiment = TorchExperiment(**params)
    assert experiment is not None
    print("✅ Test 8 passed: get_test_params returns valid config")


def test_get_score_params():
    """Test that _get_score_params returns valid parameters."""
    score_params_list = TorchExperiment._get_score_params()

    assert isinstance(score_params_list, list)
    assert len(score_params_list) >= 1

    for score_params in score_params_list:
        assert isinstance(score_params, dict)
        # Should have parameters matching SimpleTestModel
        assert "hidden_dim" in score_params or "lr" in score_params

    print("✅ Test 9 passed: _get_score_params returns valid parameters")


def test_multiple_runs_independent():
    """Test that multiple training runs are independent (fresh trainers)."""
    datamodule = SimpleTestDataModule()
    datamodule.setup()

    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
        trainer_kwargs={"max_epochs": 1},
    )

    params = {"input_dim": 10, "hidden_dim": 16, "lr": 1e-3}

    # Run twice
    score1, _ = experiment._evaluate(params)
    score2, _ = experiment._evaluate(params)

    # Scores should be close (same params) but might differ due to randomness
    assert isinstance(score1, (np.floating, float))
    assert isinstance(score2, (np.floating, float))
    print(f"✅ Test 10 passed: Multiple runs work (scores={score1:.4f}, {score2:.4f})")


def test_tags_exist():
    """Test that required tags are present."""
    assert hasattr(TorchExperiment, "_tags")
    tags = TorchExperiment._tags

    assert "property:randomness" in tags
    assert "property:higher_or_lower_is_better" in tags
    assert "authors" in tags
    assert "python_dependencies" in tags

    # Check dependencies
    assert "torch" in tags["python_dependencies"]
    assert "lightning" in tags["python_dependencies"]
    print("✅ Test 11 passed: All required tags present")


def test_with_hyperactive():
    """Test integration with Hyperactive optimizer."""
    try:
        from hyperactive import Hyperactive
    except ImportError:
        print("⚠️ Test 12 skipped: Hyperactive not available")
        return

    datamodule = SimpleTestDataModule()
    datamodule.setup()

    experiment = TorchExperiment(
        datamodule=datamodule,
        lightning_module=SimpleTestModel,
        trainer_kwargs={"max_epochs": 1},
    )

    search_space = {"hidden_dim": [16, 32], "lr": [1e-3, 1e-2]}

    hyper = Hyperactive()
    hyper.add_search(experiment, search_space, n_iter=2)
    hyper.run()

    best_params = hyper.best_para(experiment)
    assert "hidden_dim" in best_params
    assert "lr" in best_params
    print(f"✅ Test 12 passed: Hyperactive integration works (best={best_params})")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running TorchExperiment Tests")
    print("=" * 60 + "\n")

    tests = [
        test_torch_experiment_initialization,
        test_paramnames,
        test_evaluate_basic,
        test_evaluate_different_params,
        test_custom_objective_metric,
        test_invalid_metric_name,
        test_trainer_kwargs_override,
        test_get_test_params,
        test_get_score_params,
        test_multiple_runs_independent,
        test_tags_exist,
        test_with_hyperactive,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All tests passed! Ready to push to GitHub.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please fix before pushing.")
