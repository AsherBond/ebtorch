# justfile for ebtorch project

# Configuration
ruff_config := "~/ruffconfigs/ebdefault/ruff.toml"

# Default recipe to display help when running just without arguments
default:
    @just --list

# Display this help message
help:
    @just --list

# Clean Python cache files and directories
clean:
    @find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
    @find . -type d \( -name ".mypy_cache" -o -name "__pycache__" -o -name ".ruff_cache" \) -exec rm -rf {} + 2>/dev/null || true

# Format Python files and sort requirements
format:
    #!/usr/bin/env bash
    set -euo pipefail
    python_files=$(find . -name "*.py" -type f)
    if [ -n "$python_files" ]; then
        for file in $python_files; do
            reorder-python-imports --py310-plus "$file" || exit 1
        done
        ruff format --config "{{ruff_config}}" . || exit 1
    fi
    requirements_files=$(find . -name "requirements.txt" -type f)
    if [ -n "$requirements_files" ]; then
        for file in $requirements_files; do
            sort-requirements "$file" || exit 1
        done
    fi

# Deploy git hooks from .githooks directory
deployhooks:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d ./.githooks ]; then
        cp -f ./.githooks/* ./.git/hooks/
        chmod +x ./.git/hooks/*
    else
        exit 1
    fi

# Run pre-commit autoupdate
precau:
    @pre-commit autoupdate

# Run pre-commit on all files
precra:
    @pre-commit run --all-files

# Check if git status is not clean (has changes)
check-git-status:
    #!/usr/bin/env bash
    if [ -z "$(git status --porcelain)" ]; then
        exit 1
    fi

# Add all changes, commit, and push
gitall: check-git-status
    @git add -A
    @git commit --all
    @git push

# Alias for precra
lint: precra

# Alias for format
fmt: format

# Prepare for git: format, update pre-commit, run checks, and clean
gitpre: format precau precra clean

# Format, update pre-commit, run checks, clean, and push to git
gitpush: format precau precra clean gitall

# Clean and format
clfmt: format clean
