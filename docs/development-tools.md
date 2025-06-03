# Development Tools Guide

This document describes the development tools used in this project and how to use them effectively.

## Code Formatters

### YAPF (Yet Another Python Formatter)
- **Purpose**: Automatically formats Python code according to style guidelines
- **Configuration**: Uses `.style.yapf` file
- **Usage**: 
  ```bash
  yapf -i filename.py  # Format a single file
  yapf -i -r .        # Format all Python files recursively
  ```

### isort
- **Purpose**: Sorts and organizes Python imports
- **Configuration**: Uses Google profile (specified in pre-commit config)
- **Usage**:
  ```bash
  isort filename.py   # Sort imports in a single file
  isort .            # Sort imports in all Python files
  ```

## Linters

### Pylint
- **Purpose**: Static code analysis to catch errors and enforce coding standards
- **Configuration**: Uses `.pylintrc` file
- **Usage**:
  ```bash
  pylint filename.py  # Lint a single file
  pylint .           # Lint all Python files
  ```

### Flake8
- **Purpose**: Style guide enforcement and error detection
- **Usage**:
  ```bash
  flake8 filename.py  # Check a single file
  flake8 .           # Check all Python files
  ```

## Type Checking

### MyPy
- **Purpose**: Static type checking for Python code
- **Configuration**: Configured in pre-commit hooks
- **Usage**:
  ```bash
  mypy filename.py   # Type check a single file
  mypy .            # Type check all Python files
  ```

## Testing

### Pytest
- **Purpose**: Testing framework for Python code
- **Location**: Tests should be placed in the `tests/` directory
- **Usage**:
  ```bash
  pytest             # Run all tests
  pytest test_file.py # Run specific test file
  pytest -v         # Run tests with verbose output
  pytest -k "test_name" # Run tests matching the given name
  ```

## Documentation

### Sphinx
- **Purpose**: Generates project documentation
- **Location**: Documentation source files are in `docs/`
- **Usage**:
  ```bash
  cd docs
  make html         # Build HTML documentation
  ```

## Pre-commit Hooks

The project uses pre-commit hooks to automatically run checks before each commit.

### Setup
```bash
pre-commit install  # Install the pre-commit hooks
```

### Available Hooks
- trailing-whitespace: Removes trailing whitespace
- end-of-file-fixer: Ensures files end with a newline
- check-yaml: Validates YAML files
- check-added-large-files: Prevents large files from being committed
- yapf: Formats Python code
- pylint: Runs linting
- mypy: Performs type checking
- isort: Sorts imports

### Manual Run
```bash
pre-commit run --all-files  # Run all hooks on all files
pre-commit run hook-name    # Run a specific hook
```

## Virtual Environment

### Setup
```bash
python -m venv venv        # Create virtual environment
source venv/bin/activate   # Activate (Linux/Mac)
.\venv\Scripts\activate    # Activate (Windows)
```

### Package Management
```bash
pip install -r requirements.txt         # Install production dependencies
pip install -r requirements-dev.txt     # Install development dependencies
``` 