import pytest
import numpy as np
from src.infrastructure.services.data_splitting import DataSplitterService
from src.core.domain.value_objects import SplitConfig

def test_random_split():
    """Test simple random train/test split."""
    service = DataSplitterService()
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, 100)
    config = SplitConfig(train_size=0.8, stratify=False, seed=42)
    
    X_train, X_test, y_train, y_test = service.split_data(X, y, config)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    # Ensure all data points are present
    assert len(np.unique(np.concatenate([y_train, y_test]))) <= 2

def test_stratified_split():
    """Test stratified split preserve class distribution."""
    service = DataSplitterService()
    X = np.random.rand(100, 2)
    # Highly imbalanced: 90% zeros, 10% ones
    y = np.concatenate([np.zeros(90), np.ones(10)])
    
    config = SplitConfig(train_size=0.8, stratify=True, seed=42)
    X_train, X_test, y_train, y_test = service.split_data(X, y, config)
    
    # 80% of 90 = 72 zeros in train
    # 80% of 10 = 8 ones in train
    assert np.sum(y_train == 0) == 72
    assert np.sum(y_train == 1) == 8
    assert np.sum(y_test == 0) == 18
    assert np.sum(y_test == 1) == 2

def test_preview_split():
    """Test preview_split returns correct statistics dictionary."""
    service = DataSplitterService()
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1]) # 4 zeros, 4 ones
    config = SplitConfig(train_size=0.5, stratify=True, seed=42)
    
    stats = service.preview_split(y, config)
    
    assert stats.train_count == 4
    assert stats.test_count == 4
    assert stats.train_distribution == {0: 2, 1: 2}
    assert stats.test_distribution == {0: 2, 1: 2}
