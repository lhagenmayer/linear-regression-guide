# Development & Testing Guide

This comprehensive guide covers development workflows, code quality standards, testing infrastructure, and contribution guidelines for the Linear Regression Guide project.

## 🚀 Quick Start Development

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd linear-regression-guide

# Automated setup (recommended)
./scripts/setup_dev.sh

# Or manual setup:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

### 2. Verify Setup

```bash
# Check code quality
./scripts/verify_code_quality.sh

# Run tests
./scripts/run_tests.sh

# Start development
streamlit run run.py
```

## 🛠️ Development Workflow

### Daily Development

```bash
# Make changes to code
# ...

# Check code quality
./scripts/verify_code_quality.sh

# Run relevant tests
./scripts/run_tests.sh --unit

# Commit changes
git add .
git commit -m "Your changes"
```

### Before Pushing

```bash
# Full test suite
./scripts/run_tests.sh --coverage

# Code quality
./scripts/verify_code_quality.sh

# Repository cleanup
./scripts/clean_repo.sh
```

## 📋 Code Quality Standards

### Automated Tools

#### Black - Code Formatter
```bash
# Format code
black src/*.py tests/*.py run.py

# Check formatting
black --check src/*.py tests/*.py run.py

# Show differences
black --diff src/*.py tests/*.py run.py
```

**Configuration**: `pyproject.toml`
- Line length: 120 characters
- Target Python: 3.9+

#### Flake8 - Linter
```bash
# Lint code
flake8 src/*.py tests/*.py run.py

# Show statistics
flake8 src/*.py tests/*.py run.py --statistics
```

**Configuration**: `.flake8`
- Max line length: 120 characters
- Complexity limit: 15
- **Ignored rules**: E203, W503, E501 (Black compatibility)

#### MyPy - Type Checker
```bash
# Type check
mypy src/app.py src/config.py src/data.py src/content.py src/plots.py src/logger.py src/accessibility.py
```

**Configuration**: `mypy.ini`
- Strict mode enabled
- Ignore missing imports for external packages

### Manual Code Review Checklist

- [ ] Functions have docstrings
- [ ] Complex logic has comments
- [ ] No hardcoded values (use config)
- [ ] Error handling for edge cases
- [ ] Logging for important operations
- [ ] Tests added for new functionality

## 🧪 Testing Infrastructure

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_data.py            # Data generation unit tests
├── test_app_integration.py # Streamlit AppTest integration tests
├── test_performance.py     # Performance regression tests
├── test_error_handling.py  # Error boundary and edge case tests
├── test_property_based.py  # Hypothesis property-based tests
├── test_visual_regression.py # Visual regression tests
└── test_accessibility.py   # Accessibility feature tests
```

### Running Tests

#### All Tests
```bash
./scripts/run_tests.sh
```

#### Specific Test Suites
```bash
# Unit tests only
./scripts/run_tests.sh --unit

# Integration tests only
./scripts/run_tests.sh --integration

# Performance tests only
./scripts/run_tests.sh --performance
```

#### Advanced Options
```bash
# With coverage
./scripts/run_tests.sh --coverage

# CI/CD mode
./scripts/run_tests.sh --ci

# Verbose output
./scripts/run_tests.sh --verbose

# Keep going on failures
./scripts/run_tests.sh --keep-going
```

### Writing Tests

#### Unit Test Example
```python
import pytest
from src.data import generate_simple_regression_data

class TestDataGeneration:
    def test_valid_data_generation(self):
        \"\"\"Test that data generation works with valid inputs.\"\"\"
        result = generate_simple_regression_data(
            '🇨🇭 Schweizer Kantone (sozioökonomisch)',
            'Population Density',
            50,
            42
        )

        assert 'x' in result
        assert 'y' in result
        assert len(result['x']) == 50
        assert len(result['y']) == 50
```

#### Integration Test Example
```python
import pytest
from streamlit.testing.v1 import AppTest

def test_app_initialization():
    \"\"\"Test that the app initializes correctly.\"\"\"
    at = AppTest.from_file("run.py")
    at.run(timeout=30)

    assert not at.exception
    assert "Linear Regression Guide" in str(at.main)
```

### Test Categories

#### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Focus on logic and edge cases

#### Integration Tests
- Test component interactions
- Use Streamlit AppTest for UI testing
- Verify end-to-end workflows

#### Performance Tests
- Test execution time and memory usage
- Verify caching effectiveness
- Monitor for performance regressions

#### Property-Based Tests
- Use Hypothesis to generate test cases
- Test mathematical properties
- Find edge cases automatically

#### Visual Regression Tests
- Verify plot generation consistency
- Check visual output signatures
- Detect rendering changes

### Coverage Goals

- **Statements**: >90%
- **Branches**: >80%
- **Functions**: >95%
- **Lines**: >90%

Generate coverage reports:
```bash
./scripts/run_tests.sh --coverage
# View HTML report in htmlcov/index.html
```

## 🤝 Contribution Guidelines

### Branching Strategy

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# Test thoroughly
# Commit with clear messages

# Push and create PR
git push origin feature/your-feature-name
```

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions
- `chore`: Maintenance

**Examples:**
```
feat(data): add Swiss weather dataset support
fix(plots): resolve 3D rendering issue on mobile
docs(readme): update deployment instructions
test(validation): add edge case tests for data generation
```

### Code Review Process

1. **Automated Checks**: All CI checks must pass
2. **Manual Review**: At least one maintainer review
3. **Testing**: New features require appropriate tests
4. **Documentation**: Update docs for user-facing changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Testing improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No breaking changes
```

## 🚀 Deployment

### Local Deployment
```bash
# Test deployment locally
./scripts/prepare_deployment.sh

# Start app
streamlit run run.py
```

### Streamlit Cloud Deployment
```bash
# Prepare deployment
./scripts/prepare_deployment.sh --deploy

# Then manually deploy via https://share.streamlit.io
```

### CI/CD Integration

GitHub Actions automatically:
- Run tests on push/PR
- Check code quality
- Validate deployment readiness
- Generate coverage reports

## 🐛 Debugging

### Common Issues

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify package structure
find . -name "*.py" | head -10
```

#### Test Failures
```bash
# Run specific test with debug
./scripts/run_tests.sh --verbose --unit

# Check test isolation
python -m pytest tests/test_data.py::TestSafeScalar::test_safe_scalar_with_float -v
```

#### Performance Issues
```bash
# Profile specific function
python -c "
import cProfile
from src.data import generate_simple_regression_data
cProfile.run('generate_simple_regression_data(\"🇨🇭 Schweizer Kantone (sozioökonomisch)\", \"Population Density\", 100, 42)')
"
```

## 📊 Project Metrics

### Current Statistics
- **Lines of Code**: ~8,000
- **Test Coverage**: >90%
- **Python Files**: 9 core modules
- **Test Files**: 8 comprehensive test suites

### Quality Metrics
- **Cyclomatic Complexity**: <15 per function
- **Maintainability Index**: >80
- **Technical Debt**: Minimal
- **Documentation Coverage**: >95%

## 📚 Additional Resources

- [Python Best Practices](https://python-guide.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Statsmodels Documentation](https://www.statsmodels.org/)

---

*This guide is maintained automatically. Last updated: January 2026*
- F541: f-string without placeholders

### MyPy - Static Type Checker

MyPy provides optional static type checking for Python code.

- **Configuration**: `mypy.ini`
- **Mode**: Lenient (does not require all functions to be typed)

```bash
# Run mypy on main modules
mypy app.py config.py data.py plots.py

# Run mypy with more verbose output
mypy --show-error-codes app.py
```

**Note**: MyPy is configured to be lenient and runs with `continue-on-error` in CI/CD. It provides helpful warnings but doesn't fail the build.

## Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit.

### Installation

```bash
# Install pre-commit hooks
pip install pre-commit

# Set up git hooks
pre-commit install
```

### Usage

```bash
# Run manually on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files file1.py file2.py

# Update hooks to latest versions
pre-commit autoupdate

# Bypass hooks (not recommended)
git commit --no-verify -m "message"
```

### Hooks Configured

1. **Trailing Whitespace**: Remove trailing whitespace
2. **End of File Fixer**: Ensure files end with a newline
3. **YAML/JSON/TOML Checker**: Validate configuration files
4. **Large File Checker**: Prevent committing large files (>1MB)
5. **Merge Conflict Checker**: Detect unresolved merge conflicts
6. **Debug Statements**: Find accidentally committed debug statements
7. **Black**: Automatic code formatting
8. **Flake8**: Style and error checking

## Development Workflow

### 1. Before Starting Work

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify everything works
pre-commit run --all-files
pytest tests/
```

### 2. During Development

```bash
# Format code frequently
black *.py tests/*.py

# Check for issues
flake8 *.py tests/*.py

# Run tests
pytest tests/ -v
```

### 3. Before Committing

```bash
# Run all checks
pre-commit run --all-files

# Run full test suite
pytest tests/ --cov

# Verify no issues
flake8 *.py tests/*.py
```

### 4. Commit

```bash
# Pre-commit hooks run automatically
git add .
git commit -m "Your descriptive message"

# If hooks fail, fix issues and try again
git add .
git commit -m "Your descriptive message"
```

## GitHub Actions CI/CD

### Workflows

1. **tests.yml**: Runs full test suite on multiple Python versions
2. **lint.yml**: Runs code quality checks (Black, Flake8, MyPy)

### Lint Workflow

The lint workflow runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual trigger via workflow_dispatch

**Checks performed**:
- Black formatting check
- Flake8 linting
- MyPy type checking (non-blocking)

## Configuration Files

### pyproject.toml

Contains configuration for:
- Black formatter
- isort (import sorting)
- Pytest

### .flake8

Flake8 configuration including:
- Max line length
- Ignored rules
- Excluded directories
- Complexity limits

### mypy.ini

MyPy configuration with:
- Python version target
- Warning levels
- Module-specific settings
- Import handling

### .pre-commit-config.yaml

Pre-commit hooks configuration including:
- Hook repositories
- Hook IDs and versions
- Arguments and exclusions

## Code Style Guidelines

### General Principles

1. **Consistency**: Follow existing code patterns
2. **Readability**: Code is read more often than written
3. **Simplicity**: Prefer simple, clear solutions
4. **Documentation**: Document complex logic
5. **Testing**: Write tests for new features

### Python-Specific

1. **Line Length**: Maximum 100 characters
2. **Imports**: Organized (stdlib, third-party, local)
3. **Naming**:
   - Functions/variables: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
4. **Docstrings**: Use for public functions and classes
5. **Type Hints**: Optional but encouraged for function signatures

### Examples

```python
# Good
def calculate_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Calculate linear regression parameters.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        Dictionary with slope and intercept
    """
    slope = np.polyfit(x, y, 1)[0]
    intercept = np.polyfit(x, y, 1)[1]
    return {"slope": slope, "intercept": intercept}


# Bad
def calcReg(x,y):
    s=np.polyfit(x,y,1)[0]
    i=np.polyfit(x,y,1)[1]
    return {"slope":s,"intercept":i}
```

## Troubleshooting

### Pre-commit Hooks Fail

```bash
# View detailed error
pre-commit run --all-files --verbose

# Fix formatting issues
black *.py tests/*.py

# Fix flake8 issues manually or with autoflake
pip install autoflake
autoflake --in-place --remove-unused-variables --remove-all-unused-imports *.py
```

### MyPy Errors

MyPy errors are informational and don't block commits. To fix:

```bash
# Add type annotations
def my_function(x: int) -> str:
    return str(x)

# Use type: ignore for complex cases
result = complex_function()  # type: ignore
```

### Tests Fail

```bash
# Run specific test
pytest tests/test_data.py::test_function_name -v

# Run with more output
pytest tests/ -vv --tb=long

# Run with pdb debugger
pytest tests/ --pdb
```

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Python Style Guide (PEP 8)](https://pep8.org/)

## Deployment

### Streamlit Cloud Deployment

This application is configured for easy deployment to Streamlit Cloud.

#### Quick Deployment

1. Push your changes to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set `app.py` as the main file
5. Deploy!

#### Configuration Files

- **`.streamlit/config.toml`**: Streamlit configuration (theme, server settings)
- **`requirements.txt`**: Python dependencies for deployment
- **No secrets required**: All data is simulated/offline

#### Local Testing Before Deployment

```bash
# Test the app locally exactly as it will run in production
streamlit run app.py

# Open browser to http://localhost:8501
# Test all features to ensure they work correctly
```

#### Deployment Checklist

Before deploying, verify:

- [ ] All tests pass: `pytest tests/`
- [ ] Code is formatted: `black --check *.py tests/*.py`
- [ ] No linting errors: `flake8 *.py tests/*.py`
- [ ] App runs locally: `streamlit run app.py`
- [ ] All features work without errors
- [ ] Performance is acceptable (check with different dataset sizes)

#### Post-Deployment

After deploying to Streamlit Cloud:

1. **Test thoroughly**: Go through all tabs and features
2. **Check logs**: Monitor Streamlit Cloud logs for errors
3. **Verify performance**: Ensure acceptable load times
4. **Test on mobile**: Check mobile responsiveness
5. **Update README**: Add live demo URL

#### Detailed Deployment Guide

See [scripts/README.md](../scripts/README.md) for deployment script instructions, including:

- Step-by-step deployment process
- Configuration options
- Troubleshooting guide
- Performance optimization tips
- Monitoring and maintenance
- Custom domain setup

#### Automatic Redeployment

Streamlit Cloud automatically redeploys when you push to your configured branch:

```bash
# Make changes
git add .
git commit -m "Update feature X"
git push origin main

# Streamlit Cloud detects the push and redeploys automatically
# Redeployment typically takes 2-5 minutes
```

#### Environment Differences

**Local vs. Cloud:**

| Aspect | Local | Streamlit Cloud |
|--------|-------|-----------------|
| Python Version | Your local version | 3.9-3.12 (configurable) |
| Resources | Your machine | Shared (free tier) |
| URL | localhost:8501 | your-app.streamlit.app |
| HTTPS | No | Yes |
| Authentication | None | Optional (Pro tier) |

**Note**: The app is designed to work identically in both environments.

#### Deployment Resources

- **Streamlit Cloud Docs**: [docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Deployment Scripts**: See [scripts/README.md](../scripts/README.md) for automated deployment tools

