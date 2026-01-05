# Development Scripts

This directory contains development and maintenance scripts for the Linear Regression Guide project. These scripts automate common development tasks, ensure code quality, and streamline the development workflow.

## 📋 Available Scripts

### 🔍 Code Quality & Verification

#### `verify_code_quality.sh`
Enhanced code quality verification script that checks linting, formatting, and type safety.

```bash
# Check all code quality aspects
./scripts/verify_code_quality.sh

# Auto-fix formatting issues
./scripts/verify_code_quality.sh --fix

# Show help
./scripts/verify_code_quality.sh --help
```

**Features:**
- ✅ Python syntax validation
- ✅ Import dependency checking
- ✅ Black code formatting verification
- ✅ Flake8 linting
- ✅ MyPy type checking (optional)
- ✅ Pre-commit hook validation
- ✅ Auto-fix capability with `--fix`
- ✅ Comprehensive error reporting

### 🚀 Deployment & Production

#### `prepare_deployment.sh`
Streamlit Cloud deployment preparation script that validates and optimizes the application for production.

```bash
# Check deployment readiness
./scripts/prepare_deployment.sh

# Prepare for deployment
./scripts/prepare_deployment.sh --deploy

# Verbose output
./scripts/prepare_deployment.sh --verbose
```

**Checks Performed:**
- ✅ Project structure validation
- ✅ Python environment compatibility
- ✅ Core dependency availability
- ✅ Streamlit Cloud requirements
- ✅ File size limits
- ✅ Deployment package preparation
- ✅ Comprehensive readiness report

### 🛠️ Development Environment

#### `setup_dev.sh`
Complete development environment setup script that installs dependencies and configures the development workflow.

```bash
# Full development setup
./scripts/setup_dev.sh

# Quick setup (skip optional tools)
./scripts/setup_dev.sh --quick

# Skip optional tools only
./scripts/setup_dev.sh --skip-optional
```

**Setup Includes:**
- ✅ Python environment validation
- ✅ Production dependencies installation
- ✅ Development dependencies installation
- ✅ Pre-commit hooks setup
- ✅ Initial code quality verification
- ✅ Development shortcuts creation
- ✅ Workflow guidance

### 🧪 Testing Suite

#### `run_tests.sh`
Comprehensive test runner that executes all test suites with coverage reporting and CI/CD integration.

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test suites
./scripts/run_tests.sh --unit
./scripts/run_tests.sh --integration
./scripts/run_tests.sh --performance
./scripts/run_tests.sh --property
./scripts/run_tests.sh --visual

# With coverage reporting
./scripts/run_tests.sh --coverage

# CI/CD mode
./scripts/run_tests.sh --ci

# Verbose output
./scripts/run_tests.sh --verbose
```

**Test Suites:**
- **Unit Tests** - Individual function testing
- **Integration Tests** - Streamlit AppTest workflows
- **Performance Tests** - Speed and resource testing
- **Property-based Tests** - Hypothesis-generated edge cases
- **Visual Tests** - Plot generation and consistency
- **Error Handling Tests** - Edge cases and validation

### 🧹 Repository Maintenance

#### `clean_repo.sh`
Repository cleanup and maintenance script that removes temporary files and optimizes repository size.

```bash
# Safe cleanup (recommended)
./scripts/clean_repo.sh

# Aggressive cleanup (removes more files)
./scripts/clean_repo.sh --aggressive

# Dry run (show what would be cleaned)
./scripts/clean_repo.sh --dry-run

# Verbose output
./scripts/clean_repo.sh --verbose
```

**Cleanup Actions:**
- **Safe Mode:**
  - Python cache files (`__pycache__`, `*.pyc`)
  - Test artifacts (`.pytest_cache`, coverage reports)
  - IDE files (`.DS_Store`, swap files)
  - Build artifacts (`build/`, `dist/`)

- **Aggressive Mode (additional):**
  - Coverage reports (`htmlcov/`, `coverage.xml`)
  - Log files (`*.log`)
  - Documentation builds (`docs/_build/`)
  - Download cache (`.cache/`)

### 📚 Documentation Generation

#### `generate_docs.sh`
Documentation generation script that creates API docs, updates README badges, and generates project statistics.

```bash
# Generate all documentation
./scripts/generate_docs.sh

# Generate API docs only
./scripts/generate_docs.sh --api-only

# Update README only
./scripts/generate_docs.sh --readme-only

# Update badges only
./scripts/generate_docs.sh --badges-only
```

**Generates:**
- ✅ API documentation (HTML via pdoc)
- ✅ README badges (Streamlit, Python version, license)
- ✅ Project statistics (lines of code, test coverage, etc.)
- ✅ Code documentation structure

## 🚀 Quick Start Development Workflow

```bash
# 1. Initial setup
./scripts/setup_dev.sh

# 2. Verify code quality
./scripts/verify_code_quality.sh

# 3. Run tests
./scripts/run_tests.sh

# 4. Prepare for deployment
./scripts/prepare_deployment.sh --deploy

# 5. Clean up when needed
./scripts/clean_repo.sh
```

## 🔧 Script Dependencies

### Required Tools
- **bash** - Shell environment
- **python3** - Python interpreter
- **pip** - Python package manager
- **git** - Version control (for some scripts)

### Optional Tools
- **pre-commit** - Git hooks (installed by setup script)
- **pdoc** - API documentation generation
- **pytest** - Test framework (installed via requirements-dev.txt)
- **black, flake8, mypy** - Code quality tools (installed via requirements-dev.txt)

## 📋 Script Maintenance

### Adding New Scripts

1. **Create the script** in the `scripts/` directory
2. **Make it executable**: `chmod +x scripts/your_script.sh`
3. **Add comprehensive help**: `--help` option
4. **Follow naming conventions**: `lowercase_with_underscores.sh`
5. **Add to this README**: Update the documentation

### Script Standards

All scripts should follow these standards:

- **Shebang**: `#!/usr/bin/env bash`
- **Error handling**: `set -e` for strict error checking
- **Help option**: `--help` or `-h` option
- **Color output**: Use consistent color variables
- **Logging functions**: `log_info`, `log_success`, `log_warning`, `log_error`
- **Verbose mode**: Support `--verbose` for detailed output
- **Directory validation**: Check if run from project root

### Testing Scripts

Test scripts manually and verify they work in different scenarios:

```bash
# Test help output
./scripts/your_script.sh --help

# Test basic functionality
./scripts/your_script.sh

# Test edge cases
./scripts/your_script.sh --verbose
./scripts/your_script.sh --dry-run  # if applicable

# Test error conditions
./scripts/your_script.sh --invalid-option
```

## 🐛 Troubleshooting

### Common Issues

**Script not found or permission denied:**
```bash
chmod +x scripts/*.sh
```

**Python dependencies missing:**
```bash
pip install -r requirements-dev.txt
```

**Pre-commit hooks not working:**
```bash
pre-commit install
```

**Tests failing:**
```bash
./scripts/run_tests.sh --verbose
```

### Getting Help

Each script provides detailed help:

```bash
./scripts/verify_code_quality.sh --help
./scripts/prepare_deployment.sh --help
./scripts/setup_dev.sh --help
./scripts/run_tests.sh --help
./scripts/clean_repo.sh --help
./scripts/generate_docs.sh --help
```

## 📊 Script Metrics

| Script | Lines | Features | Dependencies |
|--------|-------|----------|--------------|
| `verify_code_quality.sh` | ~150 | 6 checks | Python tools |
| `prepare_deployment.sh` | ~200 | 10 validations | Streamlit, Python |
| `setup_dev.sh` | ~250 | 8 setup steps | All dev tools |
| `run_tests.sh` | ~180 | 5 test suites | pytest, coverage |
| `clean_repo.sh` | ~220 | 2 cleanup modes | find, rm |
| `generate_docs.sh` | ~150 | 3 doc types | pdoc, Python |

## 🤝 Contributing

When contributing new scripts:

1. Follow the established patterns and standards
2. Include comprehensive help and error handling
3. Test on multiple platforms (macOS, Linux)
4. Update this README with documentation
5. Ensure scripts are idempotent (can be run multiple times safely)

---

**Scripts Version**: 1.0
**Last Updated**: January 2026
**Maintained by**: Linear Regression Guide Team