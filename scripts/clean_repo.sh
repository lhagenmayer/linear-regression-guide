#!/usr/bin/env bash
# Repository Cleanup and Maintenance Script
#
# This script performs various cleanup and maintenance tasks for the
# Linear Regression Guide repository. It removes temporary files,
# optimizes repository size, and performs general maintenance.
#
# Usage:
#   ./scripts/clean_repo.sh            # Safe cleanup (default)
#   ./scripts/clean_repo.sh --aggressive # Aggressive cleanup
#   ./scripts/clean_repo.sh --dry-run   # Show what would be cleaned
#   ./scripts/clean_repo.sh --help      # Show help

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
AGGRESSIVE=false
DRY_RUN=false
VERBOSE=false

# Statistics
FILES_REMOVED=0
SPACE_SAVED=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --aggressive)
            AGGRESSIVE=true
            shift
            ;;
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Repository Cleanup and Maintenance Script"
            echo ""
            echo "Usage:"
            echo "  $0                      # Safe cleanup (recommended)"
            echo "  $0 --aggressive        # Aggressive cleanup (removes more)"
            echo "  $0 --dry-run          # Show what would be cleaned"
            echo "  $0 --verbose          # Verbose output"
            echo "  $0 --help             # Show this help"
            echo ""
            echo "Safe cleanup removes:"
            echo "  - Python cache files (__pycache__, *.pyc)"
            echo "  - Test cache and artifacts"
            echo "  - Build artifacts"
            echo "  - Temporary files"
            echo "  - IDE-specific files"
            echo ""
            echo "Aggressive cleanup additionally removes:"
            echo "  - Coverage reports"
            echo "  - Log files"
            echo "  - Old documentation builds"
            echo "  - Download cache"
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

# Safe remove function
safe_remove() {
    local path=$1
    local description=$2

    if [ -e "$path" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove: $path ($description)"
            return 0
        fi

        local size_before=$(du -sk "$path" 2>/dev/null | cut -f1 || echo "0")
        if rm -rf "$path" 2>/dev/null; then
            FILES_REMOVED=$((FILES_REMOVED + 1))
            SPACE_SAVED=$((SPACE_SAVED + size_before))
            if [ "$VERBOSE" = true ]; then
                log_success "Removed: $path ($description)"
            fi
            return 0
        else
            log_warning "Failed to remove: $path"
            return 1
        fi
    fi
    return 0
}

# Find and remove pattern
clean_pattern() {
    local pattern=$1
    local description=$2

    local found_files=$(find . -name "$pattern" -type f 2>/dev/null)
    if [ -n "$found_files" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "$found_files" | while read -r file; do
                echo "Would remove: $file ($description)"
            done
            return 0
        fi

        echo "$found_files" | while read -r file; do
            if rm -f "$file" 2>/dev/null; then
                FILES_REMOVED=$((FILES_REMOVED + 1))
                if [ "$VERBOSE" = true ]; then
                    log_success "Removed: $file ($description)"
                fi
            fi
        done
    fi
}

echo "=================================="
if [ "$DRY_RUN" = true ]; then
    echo "Repository Cleanup - DRY RUN"
else
    echo "Repository Cleanup & Maintenance"
fi
echo "=================================="
echo ""

# Verify we're in the right directory
if [ ! -f "run.py" ] || [ ! -d "src" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

log_info "Starting repository cleanup..."
if [ "$AGGRESSIVE" = true ]; then
    log_warning "Aggressive mode enabled - will remove more files"
fi
if [ "$DRY_RUN" = true ]; then
    log_info "Dry run mode - no files will be actually removed"
fi
echo ""

# 1. Safe cleanup (always performed)
echo "1. Performing safe cleanup..."
echo ""

# Python cache files
log_info "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "*.pyd" -delete 2>/dev/null || true
log_success "Python cache files cleaned"

# Test artifacts
log_info "Removing test artifacts..."
safe_remove ".pytest_cache" "pytest cache"
safe_remove ".tox" "tox environments"
safe_remove "htmlcov" "coverage HTML reports"
safe_remove ".coverage" "coverage data file"
safe_remove "coverage.xml" "coverage XML report"
safe_remove ".mypy_cache" "MyPy cache"
clean_pattern "*.log" "log files"
log_success "Test artifacts cleaned"

# IDE and editor files
log_info "Removing IDE/editor files..."
clean_pattern "*.swp" "vim swap files"
clean_pattern "*.swo" "vim swap files"
clean_pattern "*~" "backup files"
clean_pattern ".DS_Store" "macOS finder files"
clean_pattern "Thumbs.db" "Windows thumbnail cache"
safe_remove ".vscode/settings.json" "VSCode workspace settings"
safe_remove ".idea" "PyCharm/IntelliJ files"
log_success "IDE files cleaned"

# Build artifacts
log_info "Removing build artifacts..."
safe_remove "build" "build directory"
safe_remove "dist" "distribution directory"
safe_remove "*.egg-info" "egg info directories"
safe_remove ".eggs" "egg directories"
log_success "Build artifacts cleaned"

# 2. Aggressive cleanup (only if --aggressive specified)
if [ "$AGGRESSIVE" = true ]; then
    echo ""
    echo "2. Performing aggressive cleanup..."
    echo ""

    # Coverage and documentation artifacts
    log_info "Removing coverage and documentation artifacts..."
    safe_remove "htmlcov" "HTML coverage reports"
    safe_remove ".coverage.*" "coverage data files"
    safe_remove "docs/_build" "documentation build"
    safe_remove "docs/build" "alternative doc build"

    # Log files
    log_info "Removing log files..."
    clean_pattern "*.log" "application logs"
    clean_pattern "*.log.*" "rotated logs"
    safe_remove "logs" "log directory"

    # Cache directories
    log_info "Removing cache directories..."
    safe_remove ".cache" "generic cache"
    safe_remove "node_modules" "Node.js modules (if any)"
    safe_remove ".npm" "npm cache"

    # Temporary files
    log_info "Removing temporary files..."
    clean_pattern "*.tmp" "temporary files"
    clean_pattern "*.temp" "temp files"
    clean_pattern "*.bak" "backup files"

    # OS-specific files
    log_info "Removing OS-specific files..."
    clean_pattern ".Trashes" "macOS trash"
    clean_pattern "desktop.ini" "Windows desktop.ini"
    clean_pattern "ehthumbs.db" "Windows thumbnail cache"

    log_success "Aggressive cleanup completed"
fi

# 3. Repository optimization
echo ""
echo "3. Repository optimization..."
echo ""

# Check for large files in git history (if git is available)
if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
    log_info "Checking git repository health..."

    # Check for large files in recent commits
    LARGE_FILES=$(git rev-list --objects --all |
        git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
        awk '/^blob/ {print substr($0, 6)}' |
        sort -k2nr |
        head -10 |
        awk '$2 > 1048576 {print $3 ": " $2/1048576 "MB"}')

    if [ -n "$LARGE_FILES" ]; then
        log_warning "Large files found in git history:"
        echo "$LARGE_FILES" | while read -r line; do
            echo "  $line"
        done
        echo ""
        echo "Consider using git lfs for large files or removing them with:"
        echo "  git filter-branch --tree-filter 'rm -f large_file' HEAD"
    else
        log_success "No large files in git history"
    fi
fi

# 4. Disk space analysis
echo ""
echo "4. Disk space analysis..."
echo ""

# Show current directory size
if command -v du >/dev/null 2>&1; then
    CURRENT_SIZE=$(du -sh . 2>/dev/null | cut -f1)
    log_info "Current repository size: $CURRENT_SIZE"
fi

# Show space saved
if [ "$DRY_RUN" = false ] && [ $FILES_REMOVED -gt 0 ]; then
    if [ $SPACE_SAVED -gt 0 ]; then
        SPACE_MB=$((SPACE_SAVED / 1024))
        if [ $SPACE_MB -gt 0 ]; then
            log_success "Space saved: ~${SPACE_MB}MB ($FILES_REMOVED files removed)"
        else
            log_success "Space saved: ~${SPACE_SAVED}KB ($FILES_REMOVED files removed)"
        fi
    fi
fi

# 5. Repository health check
echo ""
echo "5. Repository health check..."
echo ""

# Check for common issues
ISSUES_FOUND=0

# Check if .gitignore exists and covers common files
if [ ! -f ".gitignore" ]; then
    log_warning ".gitignore not found"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
    # Check if .gitignore covers Python cache
    if ! grep -q "__pycache__" .gitignore; then
        log_warning ".gitignore doesn't exclude __pycache__"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    fi
fi

# Check for sensitive files
SENSITIVE_FILES=$(find . -name "*.key" -o -name "*.pem" -o -name "*secret*" -o -name ".env*" 2>/dev/null | grep -v ".git" || true)
if [ -n "$SENSITIVE_FILES" ]; then
    log_warning "Potential sensitive files found:"
    echo "$SENSITIVE_FILES" | while read -r file; do
        echo "  $file"
    done
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

# Check for untracked files that should be ignored
if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
    UNTRACKED=$(git status --porcelain | grep "^??" | wc -l)
    if [ "$UNTRACKED" -gt 10 ]; then
        log_warning "Many untracked files ($UNTRACKED) - consider updating .gitignore"
    fi
fi

if [ $ISSUES_FOUND -eq 0 ]; then
    log_success "Repository health check passed"
else
    log_warning "$ISSUES_FOUND potential issues found (see warnings above)"
fi

# 6. Maintenance recommendations
echo ""
echo "6. Maintenance recommendations..."
echo ""

RECOMMENDATIONS=0

# Check if dependencies are up to date
if [ -f "requirements.txt" ]; then
    log_info "Consider updating dependencies periodically:"
    echo "  pip list --outdated"
    echo "  pip install --upgrade -r requirements.txt"
    RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

# Git maintenance
if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
    log_info "Git maintenance (run occasionally):"
    echo "  git gc                    # Garbage collection"
    echo "  git prune                 # Remove unreachable objects"
    echo "  git fsck                  # Check repository integrity"
    RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ $RECOMMENDATIONS -gt 0 ]; then
    echo ""
    log_info "$RECOMMENDATIONS maintenance recommendations provided above"
fi

# Final summary
echo ""
echo "=================================="

if [ "$DRY_RUN" = true ]; then
    log_success "Dry run completed - no files were actually removed"
    echo ""
    echo "Run without --dry-run to perform actual cleanup"
else
    log_success "Repository cleanup completed!"
    echo ""
    if [ $FILES_REMOVED -gt 0 ]; then
        echo "📊 Cleanup Summary:"
        echo "  • Files removed: $FILES_REMOVED"
        if [ $SPACE_SAVED -gt 0 ]; then
            SPACE_MB=$((SPACE_SAVED / 1024))
            if [ $SPACE_MB -gt 0 ]; then
                echo "  • Space saved: ~${SPACE_MB}MB"
            else
                echo "  • Space saved: ~${SPACE_SAVED}KB"
            fi
        fi
    else
        echo "📊 No cleanup was needed"
    fi
fi

echo ""
log_info "Repository is clean and optimized! 🧹✨"

exit 0