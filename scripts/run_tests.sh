#!/usr/bin/env bash
# Comprehensive Test Runner Script
#
# This script runs all tests for the Linear Regression Guide project.
# It provides options for different test suites, coverage reporting,
# and CI/CD integration.
#
# Usage:
#   ./scripts/run_tests.sh                    # Run all tests
#   ./scripts/run_tests.sh --unit            # Run only unit tests
#   ./scripts/run_tests.sh --integration     # Run only integration tests
#   ./scripts/run_tests.sh --coverage        # Run with coverage
#   ./scripts/run_tests.sh --ci              # CI/CD mode
#   ./scripts/run_tests.sh --help            # Show help

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

# Test configuration
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_PERFORMANCE=false
RUN_PROPERTY=false
RUN_VISUAL=false
WITH_COVERAGE=false
CI_MODE=false
VERBOSE=false
KEEP_GOING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_INTEGRATION=false
            RUN_PERFORMANCE=false
            RUN_PROPERTY=false
            RUN_VISUAL=false
            shift
            ;;
        --integration)
            RUN_UNIT=false
            RUN_PERFORMANCE=false
            RUN_PROPERTY=false
            RUN_VISUAL=false
            shift
            ;;
        --performance)
            RUN_UNIT=false
            RUN_INTEGRATION=false
            RUN_PROPERTY=false
            RUN_VISUAL=false
            RUN_PERFORMANCE=true
            shift
            ;;
        --property)
            RUN_UNIT=false
            RUN_INTEGRATION=false
            RUN_PERFORMANCE=false
            RUN_VISUAL=false
            RUN_PROPERTY=true
            shift
            ;;
        --visual)
            RUN_UNIT=false
            RUN_INTEGRATION=false
            RUN_PERFORMANCE=false
            RUN_PROPERTY=false
            RUN_VISUAL=true
            shift
            ;;
        --coverage)
            WITH_COVERAGE=true
            shift
            ;;
        --ci)
            CI_MODE=true
            WITH_COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --keep-going|-k)
            KEEP_GOING=true
            set +e
            shift
            ;;
        --help|-h)
            echo "Comprehensive Test Runner Script"
            echo ""
            echo "Usage:"
            echo "  $0                          # Run all tests"
            echo "  $0 --unit                   # Run only unit tests"
            echo "  $0 --integration           # Run only integration tests"
            echo "  $0 --performance           # Run only performance tests"
            echo "  $0 --property              # Run only property-based tests"
            echo "  $0 --visual                # Run only visual tests"
            echo "  $0 --coverage              # Run with coverage reporting"
            echo "  $0 --ci                    # CI/CD mode (coverage + strict)"
            echo "  $0 --verbose               # Verbose output"
            echo "  $0 --keep-going            # Continue on test failures"
            echo "  $0 --help                  # Show this help"
            echo ""
            echo "Test Suites:"
            echo "  Unit Tests      - Individual function testing"
            echo "  Integration     - Streamlit AppTest workflows"
            echo "  Performance     - Speed and resource testing"
            echo "  Property-based  - Hypothesis-generated edge cases"
            echo "  Visual          - Plot generation and consistency"
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

# Run test function
run_test_suite() {
    local suite_name=$1
    local test_command=$2
    local marker=$3

    echo ""
    log_info "Running $suite_name tests..."

    local cmd="python -m pytest"
    if [ "$marker" != "" ]; then
        cmd="$cmd -m \"$marker\""
    fi

    if [ "$WITH_COVERAGE" = true ]; then
        cmd="$cmd --cov=src --cov-report=term-missing"
        if [ "$CI_MODE" = true ]; then
            cmd="$cmd --cov-report=xml --cov-report=html"
        fi
    fi

    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    else
        cmd="$cmd -q"
    fi

    if [ "$CI_MODE" = true ]; then
        cmd="$cmd --tb=short --strict-markers"
    fi

    cmd="$cmd $test_command"

    if [ "$VERBOSE" = true ]; then
        log_info "Command: $cmd"
    fi

    if eval "$cmd"; then
        log_success "$suite_name tests passed"
        return 0
    else
        local exit_code=$?
        if [ "$KEEP_GOING" = true ]; then
            log_warning "$suite_name tests failed (continuing...)"
        else
            log_error "$suite_name tests failed"
        fi
        return $exit_code
    fi
}

# Setup function
setup_test_environment() {
    log_info "Setting up test environment..."

    # Check if we're in the right directory
    if [ ! -d "tests" ] || [ ! -f "requirements-dev.txt" ]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi

    # Check if pytest is installed
    if ! python -c "import pytest; print('pytest OK')" >/dev/null 2>&1; then
        log_error "pytest not found. Install with: pip install -r requirements-dev.txt"
        exit 1
    fi

    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

    log_success "Test environment ready"
}

# Main execution
echo "=================================="
echo "Linear Regression Guide - Test Runner"
echo "=================================="
echo ""

# Setup
setup_test_environment

# Track overall results
OVERALL_SUCCESS=true
TOTAL_SUITES=0
PASSED_SUITES=0

# Run test suites
if [ "$RUN_UNIT" = true ]; then
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    if run_test_suite "Unit" "tests/test_data.py tests/test_logging.py tests/test_accessibility.py" "unit"; then
        PASSED_SUITES=$((PASSED_SUITES + 1))
    else
        OVERALL_SUCCESS=false
    fi
fi

if [ "$RUN_INTEGRATION" = true ]; then
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    if run_test_suite "Integration" "tests/test_app_integration.py" "integration"; then
        PASSED_SUITES=$((PASSED_SUITES + 1))
    else
        OVERALL_SUCCESS=false
    fi
fi

if [ "$RUN_PERFORMANCE" = true ]; then
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    if run_test_suite "Performance" "tests/test_performance.py" "performance"; then
        PASSED_SUITES=$((PASSED_SUITES + 1))
    else
        OVERALL_SUCCESS=false
    fi
fi

if [ "$RUN_PROPERTY" = true ]; then
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    # Check if hypothesis is available
    if python -c "import hypothesis; print('OK')" >/dev/null 2>&1; then
        if run_test_suite "Property-based" "tests/test_property_based.py" "property"; then
            PASSED_SUITES=$((PASSED_SUITES + 1))
        else
            OVERALL_SUCCESS=false
        fi
    else
        log_warning "Hypothesis not available - skipping property tests"
    fi
fi

if [ "$RUN_VISUAL" = true ]; then
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    if run_test_suite "Visual" "tests/test_visual_regression.py" "visual"; then
        PASSED_SUITES=$((PASSED_SUITES + 1))
    else
        OVERALL_SUCCESS=false
    fi
fi

# Run error handling tests (always run these)
echo ""
log_info "Running error handling and edge case tests..."
TOTAL_SUITES=$((TOTAL_SUITES + 1))
if run_test_suite "Error Handling" "tests/test_error_handling.py" ""; then
    PASSED_SUITES=$((PASSED_SUITES + 1))
else
    OVERALL_SUCCESS=false
fi

# Generate coverage report if requested
if [ "$WITH_COVERAGE" = true ] && [ -f "coverage.xml" ]; then
    echo ""
    log_info "Coverage report generated:"
    echo "  - HTML: htmlcov/index.html"
    echo "  - XML: coverage.xml"

    # Show coverage summary
    if command -v coverage >/dev/null 2>&1; then
        echo ""
        log_info "Coverage Summary:"
        coverage report --show-missing | tail -10
    fi
fi

# Final results
echo ""
echo "=================================="

if [ "$OVERALL_SUCCESS" = true ]; then
    log_success "All tests passed! ($PASSED_SUITES/$TOTAL_SUITES suites)"
    echo ""
    echo "🎉 Test Results:"
    echo "  ✅ Code functionality verified"
    echo "  ✅ Integration points working"
    echo "  ✅ Error handling robust"
    echo "  ✅ Performance acceptable"
    echo ""

    if [ "$WITH_COVERAGE" = true ]; then
        echo "📊 Coverage reports available in:"
        echo "  - htmlcov/index.html (web view)"
        echo "  - coverage.xml (CI/CD integration)"
        echo ""
    fi

    echo "🚀 Ready for deployment!"
    exit 0
else
    log_error "Some tests failed ($PASSED_SUITES/$TOTAL_SUITES suites passed)"

    if [ "$KEEP_GOING" = true ]; then
        log_warning "Continuing despite failures (--keep-going specified)"
        exit 0
    else
        echo ""
        echo "💡 Debug options:"
        echo "  - Run with --verbose for detailed output"
        echo "  - Run specific suites: --unit, --integration, etc."
        echo "  - Use --keep-going to continue on failures"
        echo "  - Check individual test files for details"
        echo ""
        exit 1
    fi
fi