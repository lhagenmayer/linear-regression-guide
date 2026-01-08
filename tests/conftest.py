import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

# Füge Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_linear_data() -> Dict[str, np.ndarray]:
    """Fixture to provide consistent linear data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    # y = 2x + 1 + noise
    y = 2 * x + 1 + np.random.normal(0, 1, 100)
    return {"x": x, "y": y}

@pytest.fixture
def sample_multiple_linear_data() -> Dict[str, np.ndarray]:
    """Fixture for multiple regression data."""
    np.random.seed(42)
    n = 100
    x1 = np.random.rand(n) * 10
    x2 = np.random.rand(n) * 5
    # y = 3x1 - 2x2 + 5 + noise
    y = 3 * x1 - 2 * x2 + 5 + np.random.normal(0, 0.5, n)
    return {"x1": x1, "x2": x2, "y": y}

@pytest.fixture
def sample_classification_data() -> Dict[str, np.ndarray]:
    """Fixture for binary classification data."""
    np.random.seed(42)
    n = 100
    # Two clusters
    X1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n // 2, 2))
    y1 = np.zeros(n // 2, dtype=int)
    X2 = np.random.normal(loc=[4, 4], scale=0.5, size=(n // 2, 2))
    y2 = np.ones(n // 2, dtype=int)
    
@pytest.fixture
def sample_scaled_data() -> Dict[str, np.ndarray]:
    """Fixture for ML tests with [0, 1] range to ensure stability."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = 1.5 * x + 0.5 + np.random.normal(0, 0.1, 100)
    return {"x": x, "y": y}
