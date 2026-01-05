#!/usr/bin/env bash
# Development Environment Setup Script
#
# This script sets up a complete development environment for the
# Linear Regression Guide project. It installs dependencies,
# configures tools, and prepares the development workflow.
#
# Usage:
#   ./scripts/setup_dev.sh         # Full setup
#   ./scripts/setup_dev.sh --help  # Show help
#   ./scripts/setup_dev.sh --quick # Skip optional tools

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
QUICK_MODE=false
SKIP_OPTIONAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            SKIP_OPTIONAL=true
            shift
            ;;
        --skip-optional)
            SKIP_OPTIONAL=true
            shift
            ;;
        --help|-h)
            echo "Development Environment Setup Script"
            echo ""
            echo "Usage:"
            echo "  $0              # Full development setup"
            echo "  $0 --quick      # Quick setup (skip optional tools)"
            echo "  $0 --skip-optional # Skip optional development tools"
            echo "  $0 --help       # Show this help"
            echo ""
            echo "This script will:"
            echo "  - Verify Python environment"
            echo "  - Install production dependencies"
            echo "  - Install development dependencies"
            echo "  - Set up pre-commit hooks"
            echo "  - Configure development tools"
            echo "  - Run initial code quality checks"
            echo "  - Provide development workflow guidance"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check command availability
check_command() {
    local cmd=$1
    local description=$2

    if command -v "$cmd" >/dev/null 2>&1; then
        log_success "$description found: $cmd"
        return 0
    else
        log_error "$description not found: $cmd"
        return 1
    fi
}

# Install Python package
install_package() {
    local package=$1
    local description=$2

    log_info "Installing $description..."
    if pip install "$package"; then
        log_success "$description installed"
    else
        log_error "Failed to install $description"
        return 1
    fi
}

echo "=================================="
echo "Development Environment Setup"
echo "=================================="
echo ""

# Verify we're in the right directory
if [ ! -f "run.py" ] || [ ! -d "src" ] || [ ! -f "requirements.txt" ]; then
    log_error "Please run this script from the project root directory"
    log_info "Current directory: $(pwd)"
    log_info "Required files: run.py, src/, requirements.txt"
    exit 1
fi

log_info "Setting up development environment for Linear Regression Guide..."
echo ""

# 1. Check system requirements
echo "1. Checking system requirements..."
echo ""

# Check Python
if ! check_command python3 "Python 3"; then
    log_error "Python 3 is required but not found"
    exit 1
fi

# Check pip
if ! check_command pip "pip"; then
    log_error "pip is required but not found"
    exit 1
fi

# Check git
if ! check_command git "Git"; then
    log_warning "Git not found - pre-commit hooks will not work"
fi

# Check basic Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
    log_success "Python $PYTHON_VERSION (compatible)"
else
    log_error "Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

echo ""

# 2. Install production dependencies
echo "2. Installing production dependencies..."
echo ""

if [ -f "requirements.txt" ]; then
    log_info "Installing from requirements.txt..."
    if pip install -r requirements.txt; then
        log_success "Production dependencies installed"
    else
        log_error "Failed to install production dependencies"
        exit 1
    fi
else
    log_error "requirements.txt not found"
    exit 1
fi

echo ""

# 3. Install development dependencies
echo "3. Installing development dependencies..."
echo ""

if [ -f "requirements-dev.txt" ]; then
    log_info "Installing from requirements-dev.txt..."
    if pip install -r requirements-dev.txt; then
        log_success "Development dependencies installed"
    else
        log_error "Failed to install development dependencies"
        exit 1
    fi
else
    log_error "requirements-dev.txt not found"
    exit 1
fi

echo ""

# 4. Verify installations
echo "4. Verifying installations..."
echo ""

# Test core imports
log_info "Testing core imports..."
if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    import streamlit as st
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import statsmodels.api as sm
    import scipy
    print('All core packages imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
"; then
    log_success "Core packages working"
else
    log_error "Core package import failed"
    exit 1
fi

# Test development tools
if [ "$SKIP_OPTIONAL" = false ]; then
    log_info "Testing development tools..."
    dev_tools_ok=true

    if ! python3 -c "import black; print('black OK')" 2>/dev/null; then
        log_warning "Black not working"
        dev_tools_ok=false
    fi

    if ! python3 -c "import flake8; print('flake8 OK')" 2>/dev/null; then
        log_warning "Flake8 not working"
        dev_tools_ok=false
    fi

    if ! python3 -c "import mypy; print('mypy OK')" 2>/dev/null; then
        log_warning "MyPy not working"
        dev_tools_ok=false
    fi

    if [ "$dev_tools_ok" = true ]; then
        log_success "Development tools working"
    fi
fi

echo ""

# 5. Set up pre-commit hooks
echo "5. Setting up pre-commit hooks..."
echo ""

if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
    if command -v pre-commit >/dev/null 2>&1; then
        log_info "Installing pre-commit hooks..."
        if pre-commit install; then
            log_success "Pre-commit hooks installed"
        else
            log_warning "Failed to install pre-commit hooks"
        fi

        if [ "$QUICK_MODE" = false ]; then
            log_info "Running initial pre-commit check..."
            if pre-commit run --all-files --quiet; then
                log_success "Pre-commit checks passed"
            else
                log_warning "Some pre-commit checks failed (run 'pre-commit run --all-files' to see details)"
            fi
        fi
    else
        log_warning "pre-commit not found - install with: pip install pre-commit"
    fi
else
    log_warning "Git repository not found - skipping pre-commit setup"
fi

echo ""

# 6. Run initial code quality check
echo "6. Running initial code quality check..."
echo ""

if [ -x "scripts/verify_code_quality.sh" ]; then
    log_info "Running code quality verification..."
    if [ "$QUICK_MODE" = true ]; then
        if scripts/verify_code_quality.sh >/dev/null 2>&1; then
            log_success "Code quality check passed"
        else
            log_warning "Code quality issues found (run './scripts/verify_code_quality.sh' for details)"
        fi
    else
        if scripts/verify_code_quality.sh; then
            log_success "Code quality check passed"
        else
            log_warning "Code quality issues found (see output above)"
        fi
    fi
else
    log_warning "Code quality script not found or not executable"
fi

echo ""

# 7. Create development shortcuts (optional)
if [ "$SKIP_OPTIONAL" = false ]; then
    echo "7. Setting up development shortcuts..."
    echo ""

    # Create local bin directory if it doesn't exist
    if [ ! -d "bin" ]; then
        mkdir -p bin
        log_success "Created bin/ directory for development scripts"
    fi

    # Create development shortcuts
    cat > bin/dev << 'EOF'
#!/bin/bash
# Development shortcuts for Linear Regression Guide

case "$1" in
    "run")
        echo "Starting Streamlit app..."
        streamlit run run.py
        ;;
    "test")
        echo "Running tests..."
        ./scripts/run_tests.sh
        ;;
    "quality")
        echo "Checking code quality..."
        ./scripts/verify_code_quality.sh
        ;;
    "deploy")
        echo "Preparing deployment..."
        ./scripts/prepare_deployment.sh --deploy
        ;;
    "clean")
        echo "Cleaning repository..."
        ./scripts/clean_repo.sh
        ;;
    "help"|*)
        echo "Development shortcuts for Linear Regression Guide"
        echo ""
        echo "Usage: ./bin/dev <command>"
        echo ""
        echo "Commands:"
        echo "  run      - Start the Streamlit app"
        echo "  test     - Run all tests"
        echo "  quality  - Check code quality"
        echo "  deploy   - Prepare for deployment"
        echo "  clean    - Clean repository"
        echo "  help     - Show this help"
        ;;
esac
EOF

    chmod +x bin/dev
    log_success "Created development shortcuts (./bin/dev)"
fi

echo ""

# 8. Final setup and instructions
echo "=================================="
log_success "Development environment setup complete!"
echo ""

echo "🎉 Your development environment is ready!"
echo ""

if [ "$QUICK_MODE" = true ]; then
    echo "🚀 Quick setup completed. Run './scripts/setup_dev.sh' later for full setup."
else
    echo "🚀 Full development environment configured."
fi

echo ""
echo "📖 Development Workflow:"
echo ""
echo "  # Start developing"
echo "  streamlit run run.py"
echo ""
echo "  # Run tests"
echo "  ./scripts/run_tests.sh"
echo ""
echo "  # Check code quality"
echo "  ./scripts/verify_code_quality.sh"
echo ""
echo "  # Auto-fix formatting"
echo "  ./scripts/verify_code_quality.sh --fix"
echo ""
echo "  # Prepare for deployment"
echo "  ./scripts/prepare_deployment.sh --deploy"
echo ""

if [ "$SKIP_OPTIONAL" = false ] && [ -f "bin/dev" ]; then
    echo "  # Use development shortcuts"
    echo "  ./bin/dev run      # Start app"
    echo "  ./bin/dev test     # Run tests"
    echo "  ./bin/dev quality  # Check quality"
    echo "  ./bin/dev deploy   # Prepare deployment"
    echo ""
fi

echo "📚 Project Structure:"
echo "  run.py              - Main entry point"
echo "  src/                - Source code"
echo "  tests/              - Test suite"
echo "  scripts/            - Development scripts"
echo "  docs/               - Documentation"
echo "  requirements.txt    - Production dependencies"
echo "  requirements-dev.txt - Development dependencies"
echo ""

echo "🎯 Next Steps:"
echo "1. Start coding: streamlit run run.py"
echo "2. Run tests: ./scripts/run_tests.sh"
echo "3. Check quality: ./scripts/verify_code_quality.sh"
echo "4. Deploy when ready: ./scripts/prepare_deployment.sh --deploy"
echo ""

log_info "Happy coding! 🚀"

exit 0