#!/usr/bin/env bash
# Enhanced Code Quality Verification Script
# Run this script to verify all linting and formatting tools work correctly
#
# Usage:
#   ./scripts/verify_code_quality.sh          # Run all checks
#   ./scripts/verify_code_quality.sh --fix    # Auto-fix formatting issues
#   ./scripts/verify_code_quality.sh --help   # Show help

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
AUTO_FIX=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            AUTO_FIX=true
            shift
            ;;
        --help|-h)
            echo "Enhanced Code Quality Verification Script"
            echo ""
            echo "Usage:"
            echo "  $0              # Run all checks"
            echo "  $0 --fix        # Auto-fix formatting issues"
            echo "  $0 --help       # Show this help"
            echo ""
            echo "This script verifies:"
            echo "  - Python syntax and imports"
            echo "  - Code formatting (Black)"
            echo "  - Code linting (Flake8)"
            echo "  - Type checking (MyPy)"
            echo "  - Pre-commit hooks"
            echo "  - Import organization"
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

echo "=================================="
echo "Enhanced Code Quality Verification"
echo "=================================="
echo ""

# Track overall status
ALL_PASSED=true
WARNINGS=false

# Function to run check and report status
run_check() {
    local name=$1
    local command=$2
    local required=$3  # "required" or "optional"

    echo -e "${BLUE}Running ${name}...${NC}"

    # Capture both stdout and stderr
    if eval "$command" > /tmp/check_output.log 2>&1; then
        echo -e "${GREEN}✓${NC} $name passed"
        return 0
    else
        local exit_code=$?
        if [ "$required" = "required" ]; then
            echo -e "${RED}✗${NC} $name failed"
            echo "   Output:"
            cat /tmp/check_output.log | head -10 | sed 's/^/     /'
            if [ $(wc -l < /tmp/check_output.log) -gt 10 ]; then
                echo "     ... (truncated)"
            fi
            ALL_PASSED=false
        else
            echo -e "${YELLOW}⚠${NC} $name has issues (informational)"
            WARNINGS=true
        fi
        return $exit_code
    fi
}

# Function to show progress
show_progress() {
    echo -e "${BLUE}$1${NC}"
}

# Check if we're in the right directory
if [ ! -f "run.py" ] || [ ! -d "src" ] || [ ! -d "tests" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if dependencies are installed
show_progress "Checking dependencies..."
deps=("black" "flake8" "mypy" "pre-commit")
missing_deps=()

for dep in "${deps[@]}"; do
    if ! command -v "$dep" &> /dev/null; then
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -ne 0 ]; then
    echo -e "${RED}✗${NC} Missing dependencies: ${missing_deps[*]}"
    echo "Run: pip install -r requirements-dev.txt"
    exit 1
fi
echo -e "${GREEN}✓${NC} All dependencies installed"
echo ""

# Python syntax and import checks
show_progress "Checking Python syntax and imports..."
echo ""

# Check Python files for syntax errors
python_files=("run.py" "src/app.py" "src/config.py" "src/data.py" "src/content.py" "src/plots.py" "src/logger.py" "src/accessibility.py")

for file in "${python_files[@]}"; do
    if [ -f "$file" ]; then
        if python -m py_compile "$file" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $file syntax OK"
        else
            echo -e "${RED}✗${NC} $file has syntax errors"
            ALL_PASSED=false
        fi
    else
        echo -e "${RED}✗${NC} $file not found"
        ALL_PASSED=false
    fi
done
echo ""

# Check imports work
if python -c "
import sys
sys.path.insert(0, 'src')
try:
    import config, data, content, plots, logger, accessibility
    print('All modules import successfully')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} All modules import successfully"
else
    echo -e "${RED}✗${NC} Module import issues"
    ALL_PASSED=false
fi
echo ""

# Code quality checks
if [ "$AUTO_FIX" = true ]; then
    show_progress "Auto-fixing code formatting..."
    echo ""
    run_check "Black auto-format" "python -m black src/*.py tests/*.py run.py scripts/*.sh" "required"
    echo ""
fi

show_progress "Running code quality checks..."
echo ""

# File patterns to check
PYTHON_FILES="src/*.py tests/*.py run.py"
SHELL_FILES="scripts/*.sh"

# Run quality checks
run_check "Black formatting" "python -m black --check --diff $PYTHON_FILES" "required"
run_check "Flake8 linting" "python -m flake8 $PYTHON_FILES --max-line-length=120 --extend-ignore=E203,W503" "required"
run_check "MyPy type checking" "python -m mypy src/app.py src/config.py src/data.py src/content.py src/plots.py src/logger.py src/accessibility.py --ignore-missing-imports" "optional"
run_check "Shell script syntax" "bash -n $SHELL_FILES" "required"

# Check if pre-commit is set up
echo ""
show_progress "Checking development setup..."

if [ -f .git/hooks/pre-commit ]; then
    echo -e "${GREEN}✓${NC} Pre-commit hooks are installed"
    run_check "Pre-commit hooks" "pre-commit run --all-files --quiet" "required"
else
    echo -e "${YELLOW}⚠${NC} Pre-commit hooks not installed (run: pre-commit install)"
    WARNINGS=true
fi

# Check for common issues
if [ -f .gitignore ]; then
    echo -e "${GREEN}✓${NC} .gitignore exists"
else
    echo -e "${YELLOW}⚠${NC} .gitignore not found"
    WARNINGS=true
fi

# Check if all required files exist
required_files=("requirements.txt" "requirements-dev.txt" "pyproject.toml" "pytest.ini" ".coveragerc")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${YELLOW}⚠${NC} $file missing"
        WARNINGS=true
    fi
done

echo ""

# Final summary
echo "=================================="
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}All required checks passed! ✓${NC}"
    echo ""
    if [ "$WARNINGS" = true ]; then
        echo -e "${YELLOW}Some informational warnings were found.${NC}"
        echo "Consider addressing them for best practices."
        echo ""
    fi

    echo "Your code meets all quality standards!"
    echo ""
    echo "Next steps:"
    echo "  - Run tests: ./scripts/run_tests.sh"
    echo "  - Deploy: ./scripts/prepare_deployment.sh"
    echo "  - Setup dev environment: ./scripts/setup_dev.sh"
    exit 0
else
    echo -e "${RED}Some required checks failed! ✗${NC}"
    echo ""
    echo "Quick fixes:"
    if [ "$AUTO_FIX" = false ]; then
        echo "  - Auto-fix formatting: $0 --fix"
    fi
    echo "  - Format code: black src/*.py tests/*.py run.py"
    echo "  - Fix linting: Review flake8 output and fix manually"
    echo "  - Install hooks: pre-commit install"
    echo "  - Check MyPy: python -m mypy src/"
    exit 1
fi
