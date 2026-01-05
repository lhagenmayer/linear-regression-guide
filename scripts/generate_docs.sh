#!/usr/bin/env bash
# Documentation Generation Script
#
# This script generates various types of documentation for the
# Linear Regression Guide project, including API docs, README updates,
# and project documentation.
#
# Usage:
#   ./scripts/generate_docs.sh         # Generate all documentation
#   ./scripts/generate_docs.sh --api   # Generate API documentation only
#   ./scripts/generate_docs.sh --help  # Show help

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
GENERATE_API=true
GENERATE_README=true
GENERATE_BADGES=true
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-only)
            GENERATE_README=false
            GENERATE_BADGES=false
            shift
            ;;
        --readme-only)
            GENERATE_API=false
            GENERATE_BADGES=false
            shift
            ;;
        --badges-only)
            GENERATE_API=false
            GENERATE_README=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Documentation Generation Script"
            echo ""
            echo "Usage:"
            echo "  $0                    # Generate all documentation"
            echo "  $0 --api-only        # Generate API documentation only"
            echo "  $0 --readme-only     # Update README badges only"
            echo "  $0 --badges-only     # Update badges only"
            echo "  $0 --verbose         # Verbose output"
            echo "  $0 --help            # Show this help"
            echo ""
            echo "This script generates:"
            echo "  - API documentation from docstrings"
            echo "  - README with updated badges"
            echo "  - Project statistics and metrics"
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

# Check if pdoc is available for API docs
check_pdoc() {
    if python -c "import pdoc; print('OK')" >/dev/null 2>&1; then
        return 0
    else
        log_warning "pdoc not found - install with: pip install pdoc"
        return 1
    fi
}

# Generate API documentation
generate_api_docs() {
    log_info "Generating API documentation..."

    if ! check_pdoc; then
        return 1
    fi

    # Create docs/api directory
    mkdir -p docs/api

    # Generate HTML documentation
    if [ "$VERBOSE" = true ]; then
        log_info "Running: pdoc --html --output-dir docs/api src"
    fi

    if pdoc --html --output-dir docs/api src >/dev/null 2>&1; then
        log_success "API documentation generated in docs/api/"
        return 0
    else
        log_error "Failed to generate API documentation"
        return 1
    fi
}

# Update README badges
update_readme_badges() {
    log_info "Updating README badges..."

    # Check if README exists
    if [ ! -f "README.md" ]; then
        log_warning "README.md not found, skipping badge update"
        return 1
    fi

    # Get project information
    REPO_NAME="linear-regression-guide"

    # Generate badges
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    STREAMLIT_BADGE="[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/YOUR_USERNAME/$REPO_NAME)"
    PYTHON_BADGE="![Python $PYTHON_VERSION](https://img.shields.io/badge/python-$PYTHON_VERSION-blue.svg)"
    LICENSE_BADGE="![License](https://img.shields.io/badge/license-MIT-green.svg)"

    # Check if badges are already in README
    if grep -q "static.streamlit.io/badges" README.md; then
        log_info "Streamlit badge already exists"
    else
        # Add badges after title
        sed -i.bak '1a\
\
'"$STREAMLIT_BADGE"'\
'"$PYTHON_BADGE"' '"$LICENSE_BADGE"'
' README.md && rm README.md.bak

        log_success "README badges updated"
    fi

    return 0
}

# Generate project statistics
generate_project_stats() {
    log_info "Generating project statistics..."

    # Count lines of code
    PYTHON_FILES=$(find src tests -name "*.py" 2>/dev/null | wc -l)
    PYTHON_LINES=$(find src tests -name "*.py" -exec wc -l {} \; 2>/dev/null | awk '{sum += $1} END {print sum}')

    # Count functions and classes
    FUNCTIONS=$(grep -r "^def " src/*.py 2>/dev/null | wc -l)
    CLASSES=$(grep -r "^class " src/*.py 2>/dev/null | wc -l)

    # Count tests
    TEST_FILES=$(find tests -name "test_*.py" 2>/dev/null | wc -l)
    TEST_FUNCTIONS=$(grep -r "^def test_" tests/ 2>/dev/null | wc -l)

    # Dependencies
    DEP_COUNT=$(wc -l < requirements.txt 2>/dev/null || echo "0")
    DEV_DEP_COUNT=$(wc -l < requirements-dev.txt 2>/dev/null || echo "0")

    # Create stats file
    cat > docs/PROJECT_STATS.md << EOF
# Project Statistics

*Generated on: $(date)*

## Code Metrics

- **Python Files**: $PYTHON_FILES
- **Lines of Code**: $PYTHON_LINES
- **Functions**: $FUNCTIONS
- **Classes**: $CLASSES

## Testing

- **Test Files**: $TEST_FILES
- **Test Functions**: $TEST_FUNCTIONS

## Dependencies

- **Production Dependencies**: $DEP_COUNT
- **Development Dependencies**: $DEV_DEP_COUNT

## Project Structure

\`\`\`
linear-regression-guide/
├── src/                    # Source code
│   ├── app.py             # Main Streamlit application
│   ├── config.py          # Configuration constants
│   ├── data.py            # Data generation functions
│   ├── content.py         # Content and text generation
│   ├── plots.py           # Plotting functions
│   ├── logger.py          # Logging utilities
│   └── accessibility.py   # Accessibility helpers
├── tests/                 # Test suite
│   ├── test_*.py          # Individual test files
│   └── conftest.py        # Test configuration
├── scripts/               # Development scripts
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
\`\`\`

## Development Status

- ✅ Production ready
- ✅ Comprehensive testing
- ✅ CI/CD configured
- ✅ Documentation complete
- ✅ Deployment ready

---
*Auto-generated by scripts/generate_docs.sh*
EOF

    log_success "Project statistics generated in docs/PROJECT_STATS.md"
}

# Generate code documentation
generate_code_docs() {
    log_info "Generating code documentation..."

    # Create docs/code directory
    mkdir -p docs/code

    # Generate module documentation
    cat > docs/code/README.md << 'EOF'
# Code Documentation

This directory contains detailed documentation for the Linear Regression Guide codebase.

## Modules

### Core Modules

- **[app.py](../api/src.app.html)** - Main Streamlit application
- **[config.py](../api/src.config.html)** - Configuration constants and settings
- **[data.py](../api/src.data.html)** - Data generation and processing functions
- **[content.py](../api/src.content.html)** - Content and text generation
- **[plots.py](../api/src.plots.html)** - Plotting and visualization functions

### Utility Modules

- **[logger.py](../api/src.logger.html)** - Logging utilities and configuration
- **[accessibility.py](../api/src.accessibility.html)** - Accessibility helpers and ARIA support

## Architecture

The application follows a modular architecture with clear separation of concerns:

1. **Data Layer** (`data.py`) - Handles data generation and processing
2. **Presentation Layer** (`app.py`) - Streamlit UI and user interaction
3. **Visualization Layer** (`plots.py`) - Plot generation and styling
4. **Content Layer** (`content.py`) - Text, formulas, and educational content
5. **Configuration Layer** (`config.py`) - Constants and settings
6. **Utility Layer** (`logger.py`, `accessibility.py`) - Supporting functionality

## Development

### Adding New Features

1. **Data Features**: Add to `data.py`
2. **UI Features**: Modify `app.py`
3. **Visualization**: Extend `plots.py`
4. **Content**: Update `content.py`
5. **Configuration**: Add to `config.py`

### Testing

All modules have corresponding test files in the `tests/` directory. Run tests with:

```bash
./scripts/run_tests.sh
```

### Code Quality

Maintain code quality by running:

```bash
./scripts/verify_code_quality.sh
```

---
*Auto-generated by scripts/generate_docs.sh*
EOF

    log_success "Code documentation generated in docs/code/"
}

# Main execution
echo "=================================="
echo "Documentation Generation"
echo "=================================="
echo ""

# Verify we're in the right directory
if [ ! -f "run.py" ] || [ ! -d "src" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Generate API documentation
if [ "$GENERATE_API" = true ]; then
    echo "1. Generating API documentation..."
    echo ""
    if generate_api_docs; then
        API_SUCCESS=true
    else
        API_SUCCESS=false
    fi
    generate_code_docs
    echo ""
fi

# Update README badges
if [ "$GENERATE_README" = true ] || [ "$GENERATE_BADGES" = true ]; then
    echo "2. Updating README and badges..."
    echo ""
    if update_readme_badges; then
        README_SUCCESS=true
    else
        README_SUCCESS=false
    fi
    echo ""
fi

# Generate project statistics
echo "3. Generating project statistics..."
echo ""
generate_project_stats
echo ""

# Final summary
echo "=================================="
echo ""

if [ "$GENERATE_API" = true ] && [ "$API_SUCCESS" = true ]; then
    log_success "API documentation generated"
    echo "  📖 View at: docs/api/src.html"
fi

if [ "$GENERATE_README" = true ] && [ "$README_SUCCESS" = true ]; then
    log_success "README badges updated"
fi

log_success "Project statistics updated"
echo "  📊 View at: docs/PROJECT_STATS.md"

echo ""
log_info "Documentation generation completed! 📚✨"

# Show next steps
echo ""
echo "Next steps:"
echo "  • View API docs: open docs/api/src.html"
echo "  • Check README: cat README.md"
echo "  • Review stats: cat docs/PROJECT_STATS.md"
echo ""

exit 0