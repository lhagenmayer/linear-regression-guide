# 🧪 Test Suite Documentation

## Overview

This comprehensive test suite ensures the reliability, correctness, and maintainability of the Linear Regression Guide application. The tests cover all aspects of the application from unit tests to integration tests, with advanced testing techniques including property-based testing and visual regression testing.

## 🏗️ Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Shared fixtures and configuration
├── test_data.py                    # Unit tests for data generation
├── test_app_integration.py         # Integration tests for Streamlit app
├── test_logging.py                 # Tests for logging functionality
├── test_accessibility.py           # Tests for accessibility features
├── test_performance.py             # Performance regression tests
├── test_error_handling.py          # Comprehensive error handling tests
├── test_property_based.py          # Property-based tests with Hypothesis
├── test_visual_regression.py       # Visual regression tests for plots
├── validate_deployment.py          # Deployment validation script
└── README.md                       # This documentation
```

## 🏃‍♂️ Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_data.py

# Run specific test class/method
pytest tests/test_data.py::TestSafeScalar::test_safe_scalar_with_float

# Run tests with coverage
pytest --cov=src --cov-report=html
```

### Advanced Test Execution

```bash
# Run only unit tests
pytest -m "unit"

# Run integration tests
pytest -m "integration"

# Run slow/performance tests
pytest --runslow -m "performance"

# Run visual regression tests
pytest -m "visual" --visual-baseline

# Run property-based tests
pytest -m "property"
```

### Test Configuration

The test suite uses several configuration files:

- **`pytest.ini`**: Main pytest configuration
- **`.coveragerc`**: Coverage reporting configuration
- **`conftest.py`**: Shared fixtures and pytest hooks

## 📊 Test Categories

### 🧩 Unit Tests (`test_data.py`, `test_logging.py`, etc.)

Test individual functions and modules in isolation.

**Coverage:**
- Data generation functions
- Utility functions (`safe_scalar`, etc.)
- Logging functionality
- Accessibility helpers
- Configuration validation

### 🔗 Integration Tests (`test_app_integration.py`)

Test complete user workflows using Streamlit's AppTest framework.

**Coverage:**
- App initialization and loading
- Tab navigation
- Widget interactions
- Session state management
- End-to-end user workflows

### ⚡ Performance Tests (`test_performance.py`)

Test performance characteristics and detect regressions.

**Coverage:**
- Data generation speed
- Plot rendering performance
- Memory usage patterns
- Caching effectiveness

### 🚨 Error Handling Tests (`test_error_handling.py`)

Comprehensive testing of error conditions and edge cases.

**Coverage:**
- Invalid input validation
- Boundary condition handling
- Exception propagation
- Resource cleanup after errors
- Numerical stability

### 🎲 Property-Based Tests (`test_property_based.py`)

Use Hypothesis to test with randomly generated inputs.

**Coverage:**
- Statistical properties of generated data
- Invariants that must always hold
- Edge cases discovered through random testing
- Reproducibility guarantees

### 👁️ Visual Regression Tests (`test_visual_regression.py`)

Ensure plot generation consistency and detect visual changes.

**Coverage:**
- Plot structure validation
- Visual consistency across runs
- Plot customization options
- Export functionality (HTML, JSON)

## 🔧 Test Fixtures

### Shared Fixtures (`conftest.py`)

- **`sample_regression_data`**: Standard regression dataset for testing
- **`sample_multiple_regression_data`**: Multiple regression test data
- **`sample_swiss_canton_data`**: Real Swiss canton data
- **`sample_config_data`**: Configuration data samples
- **`mock_logger`**: Mock logger for testing
- **`temp_data_dir`**: Temporary directory for file-based tests

### Custom Markers

- **`@pytest.mark.unit`**: Unit tests
- **`@pytest.mark.integration`**: Integration tests
- **`@pytest.mark.streamlit`**: Streamlit AppTest framework tests
- **`@pytest.mark.performance`**: Performance tests
- **`@pytest.mark.slow`**: Slow-running tests
- **`@pytest.mark.visual`**: Visual regression tests
- **`@pytest.mark.property`**: Property-based tests

## 📈 Code Coverage

The test suite aims for comprehensive code coverage:

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# View detailed HTML report
open htmlcov/index.html
```

**Coverage Goals:**
- **Statements**: >90%
- **Branches**: >80%
- **Functions**: >95%
- **Lines**: >90%

## 🔄 Continuous Integration

### GitHub Actions Workflow

The test suite runs automatically on:

- **Push to master/main/develop**
- **Pull requests**
- **Manual trigger**

**CI Pipeline:**
1. **Linting**: Black, Flake8, MyPy
2. **Unit Tests**: All test categories except slow/visual
3. **Integration Tests**: Streamlit AppTest
4. **Coverage Report**: Generated and uploaded
5. **Deployment Validation**: Pre-deployment checks

### Local Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run full test suite locally
pytest --runslow --cov=src

# Run linting
black --check src tests
flake8 src tests
mypy src
```

## 🐛 Debugging Tests

### Common Issues

1. **Plotly Import Errors**: Ensure plotly is installed (`pip install plotly`)
2. **Streamlit Context Errors**: Tests run without Streamlit server context
3. **Random Seed Issues**: Use `reset_random_seed` fixture for reproducibility
4. **Performance Test Timeouts**: Use `--runslow` flag for performance tests

### Debugging Commands

```bash
# Debug specific test
pytest tests/test_data.py -v -s --pdb

# Run tests with detailed output
pytest -v --tb=long

# Profile test performance
pytest --durations=10
```

## 📝 Test Best Practices

### Writing New Tests

1. **Use descriptive test names**: `test_function_should_handle_edge_case`
2. **Test one thing per test**: Each test should have a single responsibility
3. **Use appropriate fixtures**: Leverage shared fixtures for common setup
4. **Include docstrings**: Document what each test verifies
5. **Use parametrize for multiple inputs**: `@pytest.mark.parametrize`

### Example Test Structure

```python
import pytest

class TestMyFeature:
    """Test suite for my feature."""

    def test_basic_functionality(self, sample_data):
        """Test basic functionality with valid inputs."""
        result = my_function(sample_data)
        assert result is not None
        assert isinstance(result, expected_type)

    @pytest.mark.parametrize("invalid_input", [
        None, "", 0, -1, [], {}
    ])
    def test_error_handling(self, invalid_input):
        """Test error handling with invalid inputs."""
        with pytest.raises(ValueError):
            my_function(invalid_input)

    @pytest.mark.slow
    def test_performance(self, large_dataset):
        """Test performance with large datasets."""
        import time
        start = time.time()
        result = my_function(large_dataset)
        duration = time.time() - start
        assert duration < 1.0  # Should complete within 1 second
```

## 🎯 Quality Metrics

### Test Quality Indicators

- **Test Coverage**: >90% statement coverage
- **Test Execution Time**: <5 minutes for full suite
- **Failure Rate**: <1% in CI/CD
- **Flakiness**: <0.1% random failures

### Maintenance

- **Review Tests**: Update tests when code changes
- **Remove Obsolete Tests**: Clean up tests for removed features
- **Add Regression Tests**: Add tests for bug fixes
- **Update Fixtures**: Keep test data current and relevant

## 📞 Support

For questions about the test suite:

1. **Check this documentation first**
2. **Review existing test examples**
3. **Run tests locally for debugging**
4. **Check CI/CD logs for failures**

## 🚀 Future Enhancements

Planned improvements to the test suite:

- **Load Testing**: Test with concurrent users
- **Cross-Browser Testing**: Selenium-based UI tests
- **Accessibility Testing**: Automated a11y checks
- **Performance Benchmarking**: Historical performance tracking
- **Mutation Testing**: Test test quality with mutation testing

---

**Last Updated**: January 2026
**Test Framework**: pytest 7.4+
**Coverage Tool**: coverage 7.0+
**Property Testing**: hypothesis 6.80+