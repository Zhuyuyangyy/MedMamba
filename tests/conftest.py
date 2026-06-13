"""
Pytest configuration and shared fixtures for MedMamba test suite.
"""

import sys
import pytest
import torch

# Ensure project root is in path
sys.path.insert(0, ".")


@pytest.fixture
def small_config():
    """Small model configuration for fast tests."""
    return {
        "d_model": 32,
        "d_state": 4,
        "n_layers": 2,
        "img_size": 32,
        "patch_size": 4,
        "num_classes": 2,
        "dropout": 0.0,
    }


@pytest.fixture
def dummy_input():
    """Standard dummy input tensor."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def single_input():
    """Single sample input tensor."""
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def device():
    """Available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"
