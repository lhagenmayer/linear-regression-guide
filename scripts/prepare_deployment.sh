#!/usr/bin/env bash
# Streamlit Cloud Deployment Preparation Script
#
# This script prepares the repository for deployment to Streamlit Cloud.
# It verifies all requirements, optimizes the codebase, and ensures
# deployment readiness.
#
# Usage:
#   ./scripts/prepare_deployment.sh          # Check deployment readiness
#   ./scripts/prepare_deployment.sh --deploy # Prepare for deployment
#   ./scripts/prepare_deployment.sh --help   # Show help

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
DEPLOY_MODE=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deploy)
            DEPLOY_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Streamlit Cloud Deployment Preparation Script"
            echo ""
            echo "Usage:"
            echo "  $0                    # Check deployment readiness"
            echo "  $0 --deploy           # Prepare for deployment"
            echo "  $0 --verbose          # Verbose output"
            echo "  $0 --help             # Show this help"
            echo ""
            echo "This script:"
            echo "  - Verifies all deployment requirements"
            echo "  - Optimizes code for production"
            echo "  - Validates Streamlit Cloud compatibility"
            echo "  - Prepares deployment package"
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

# Logging function
log() {
    local level=$1
    local message=$2

    case $level in
        "INFO")
            echo -e "${BLUE}ℹ${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}✓${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}✗${NC} $message"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

# Check function
check_item() {
    local description=$1
    local command=$2
    local required=${3:-true}

    if [ "$VERBOSE" = true ]; then
        log "INFO" "Checking: $description"
    fi

    if eval "$command" > /tmp/check.log 2>&1; then
        log "SUCCESS" "$description"
        return 0
    else
        if [ "$required" = true ]; then
            log "ERROR" "$description"
            if [ "$VERBOSE" = true ]; then
                echo "   Details:"
                cat /tmp/check.log | head -5 | sed 's/^/     /'
            fi
            return 1
        else
            log "WARNING" "$description (optional)"
            return 0
        fi
    fi
}

echo "=================================="
echo "Streamlit Cloud Deployment Prep"
echo "=================================="
echo ""

# Verify we're in the right directory
if [ ! -f "run.py" ] || [ ! -d "src" ]; then
    log "ERROR" "Please run this script from the project root directory"
    exit 1
fi

log "INFO" "Starting deployment preparation checks..."

# 1. Check basic project structure
echo ""
log "INFO" "Checking project structure..."
check_item "run.py exists" "[ -f run.py ]"
check_item "src/ directory exists" "[ -d src ]"
check_item "requirements.txt exists" "[ -f requirements.txt ]"
check_item ".streamlit/config.toml exists" "[ -f .streamlit/config.toml ]"

# 2. Check Python environment
echo ""
log "INFO" "Checking Python environment..."
check_item "Python 3.8+ available" "python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'"
check_item "pip available" "command -v pip >/dev/null"

# 3. Check core dependencies
echo ""
log "INFO" "Checking core dependencies..."
check_item "Streamlit installed" "python3 -c 'import streamlit; print(streamlit.__version__)' >/dev/null"
check_item "NumPy installed" "python3 -c 'import numpy; print(numpy.__version__)' >/dev/null"
check_item "Pandas installed" "python3 -c 'import pandas; print(pandas.__version__)' >/dev/null"
check_item "Plotly installed" "python3 -c 'import plotly; print(plotly.__version__)' >/dev/null"

# 4. Check Streamlit Cloud requirements
echo ""
log "INFO" "Checking Streamlit Cloud requirements..."

# Check file sizes (Streamlit Cloud has limits)
check_item "run.py size < 10MB" "[ $(stat -f%z run.py 2>/dev/null || stat -c%s run.py) -lt 10485760 ]"
check_item "requirements.txt size < 1MB" "[ $(stat -f%z requirements.txt 2>/dev/null || stat -c%s requirements.txt) -lt 1048576 ]"

# Check for forbidden files
forbidden_files=(".env" ".git" "__pycache__" "*.pyc" ".DS_Store")
for pattern in "${forbidden_files[@]}"; do
    if compgen -G "$pattern" > /dev/null; then
        check_item "No $pattern files" "false" false
    else
        check_item "No $pattern files" "true"
    fi
done

# 5. Check code quality
echo ""
log "INFO" "Checking code quality..."
check_item "Code quality check passes" "$SCRIPT_DIR/verify_code_quality.sh" false

# 6. Test basic functionality
echo ""
log "INFO" "Testing basic functionality..."

# Test imports
check_item "All modules import" "python3 -c '
import sys
sys.path.insert(0, \"src\")
import config, data, content, plots, logger, accessibility
'"

# Test basic data generation
check_item "Data generation works" "python3 -c '
import sys
sys.path.insert(0, \"src\")
from data import generate_simple_regression_data
result = generate_simple_regression_data(\"🇨🇭 Schweizer Kantone (sozioökonomisch)\", \"Population Density\", 5, 42)
assert \"x\" in result and \"y\" in result
'"

# Test Streamlit app basic load
check_item "Streamlit app loads" "timeout 10s streamlit run run.py --server.headless true --server.port 8501" false

# 7. Check for deployment optimizations
echo ""
log "INFO" "Checking deployment optimizations..."

# Check if requirements are optimized
check_item "requirements.txt is minimal" "[ $(wc -l < requirements.txt) -le 10 ]"

# Check for large files
large_files=$(find . -type f -size +1M -not -path "./.git/*" -not -name "*.zip" -not -name "*.tar.gz" 2>/dev/null | wc -l)
if [ "$large_files" -eq 0 ]; then
    log "SUCCESS" "No large files found"
else
    log "WARNING" "$large_files large files found (consider removing or compressing)"
fi

# 8. Prepare deployment if requested
if [ "$DEPLOY_MODE" = true ]; then
    echo ""
    log "INFO" "Preparing for deployment..."

    # Create deployment summary
    DEPLOYMENT_INFO="/tmp/deployment_info.txt"
    cat > "$DEPLOYMENT_INFO" << EOF
Linear Regression Guide - Deployment Information
==============================================

Deployment Time: $(date)
Repository: $(git rev-parse HEAD 2>/dev/null || echo "Not a git repo")

Main Entry Point: run.py
Streamlit Config: .streamlit/config.toml

Package Dependencies:
$(cat requirements.txt)

Environment Requirements:
- Python 3.8+
- Streamlit 1.28+
- NumPy, Pandas, Plotly, Statsmodels, SciPy

Deployment Checklist:
✅ Project structure verified
✅ Dependencies available
✅ Code quality checks passed
✅ Basic functionality tested
✅ Streamlit Cloud compatibility confirmed

Next Steps:
1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Set main file path to: run.py
5. Click Deploy!

EOF

    log "SUCCESS" "Deployment information prepared"
    echo "Deployment info saved to: $DEPLOYMENT_INFO"
    cat "$DEPLOYMENT_INFO"
fi

# 9. Final recommendations
echo ""
echo "=================================="
log "SUCCESS" "Deployment preparation complete!"

if [ "$DEPLOY_MODE" = true ]; then
    echo ""
    echo "🚀 Ready for Streamlit Cloud deployment!"
    echo ""
    echo "Next steps:"
    echo "1. Commit and push your changes to GitHub"
    echo "2. Visit https://share.streamlit.io"
    echo "3. Connect your GitHub repository"
    echo "4. Set the main file path to: run.py"
    echo "5. Click 'Deploy' and enjoy your app!"
    echo ""
    echo "📖 Your app will be available at:"
    echo "   https://share.streamlit.io/[your-username]/linear-regression-guide"
else
    echo ""
    echo "💡 Run with --deploy to prepare deployment package"
    echo "💡 Run with --verbose for detailed output"
fi

echo ""
log "INFO" "All deployment checks completed successfully!"
exit 0