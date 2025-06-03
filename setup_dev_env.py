#!/usr/bin/env python3
"""
Setup script for development environment.
Creates virtual environment, installs dependencies, and sets up development tools.
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

def run_command(command: str, cwd: str = None) -> None:
    """Run a shell command and print output."""
    try:
        subprocess.run(command, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e}")
        sys.exit(1)

def create_virtual_environment() -> None:
    """Create and activate virtual environment."""
    print("\n=== Creating Virtual Environment ===")
    venv_path = Path("venv")
    if not venv_path.exists():
        venv.create(venv_path, with_pip=True)
        print("Virtual environment created successfully")
    else:
        print("Virtual environment already exists")

def install_dependencies() -> None:
    """Install project dependencies."""
    print("\n=== Installing Dependencies ===")
    
    # Create requirements.txt if it doesn't exist
    if not Path("requirements.txt").exists():
        with open("requirements.txt", "w") as f:
            f.write("fitparse>=1.2.0\n")
            f.write("pandas>=2.0.0\n")
            f.write("matplotlib>=3.7.0\n")
            f.write("numpy>=1.24.0\n")
    
    # Create requirements-dev.txt if it doesn't exist
    if not Path("requirements-dev.txt").exists():
        with open("requirements-dev.txt", "w") as f:
            f.write("-r requirements.txt\n")
            f.write("yapf>=0.40.1\n")
            f.write("black>=23.0.0\n")
            f.write("autopep8>=2.0.0\n")
            f.write("pylint>=3.0.3\n")
            f.write("flake8>=6.0.0\n")
            f.write("mypy>=1.8.0\n")
            f.write("pytest>=7.0.0\n")
            f.write("sphinx>=7.0.0\n")
            f.write("pre-commit>=3.5.0\n")

    # Install dependencies using the appropriate pip command
    pip_cmd = str(Path("venv/Scripts/pip") if sys.platform == "win32" else "venv/bin/pip")
    run_command(f"{pip_cmd} install -r requirements-dev.txt")

def setup_git_hooks() -> None:
    """Set up pre-commit hooks."""
    print("\n=== Setting up Git Hooks ===")
    
    # Create .pre-commit-config.yaml if it doesn't exist
    if not Path(".pre-commit-config.yaml").exists():
        with open(".pre-commit-config.yaml", "w") as f:
            f.write("""repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/google/yapf
    rev: v0.40.1
    hooks:
    -   id: yapf
        args: [--style=.style.yapf, --parallel]

-   repo: https://github.com/PyCQA/pylint
    rev: v3.0.3
    hooks:
    -   id: pylint
        args: [--rcfile=.pylintrc]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-all]

-   repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
    -   id: isort
        args: [--profile=google]
""")

    # Create .style.yapf if it doesn't exist
    if not Path(".style.yapf").exists():
        with open(".style.yapf", "w") as f:
            f.write("""[style]
based_on_style = google
column_limit = 100
indent_width = 4
""")

    # Create .pylintrc if it doesn't exist
    if not Path(".pylintrc").exists():
        run_command("pylint --generate-rcfile > .pylintrc", cwd=os.getcwd())

    # Initialize pre-commit
    pre_commit_cmd = str(Path("venv/Scripts/pre-commit") if sys.platform == "win32" else "venv/bin/pre-commit")
    run_command(f"{pre_commit_cmd} install")

def setup_documentation() -> None:
    """Set up Sphinx documentation."""
    print("\n=== Setting up Documentation ===")
    
    # Create docs directory if it doesn't exist
    docs_dir = Path("docs")
    if not docs_dir.exists():
        docs_dir.mkdir()
        
        # Initialize Sphinx
        sphinx_quickstart = str(Path("venv/Scripts/sphinx-quickstart") if sys.platform == "win32" else "venv/bin/sphinx-quickstart")
        run_command(f"{sphinx_quickstart} -q -p 'Cycling Analytics' -a 'Your Name' -v 1.0 -r 1.0 -l en --ext-autodoc --ext-viewcode --makefile --batchfile docs")

def create_directory_structure() -> None:
    """Create basic directory structure."""
    print("\n=== Creating Directory Structure ===")
    directories = [
        "tests",
        "csv-files",
        "png-files",
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main setup function."""
    print("=== Starting Development Environment Setup ===")
    
    create_directory_structure()
    create_virtual_environment()
    install_dependencies()
    setup_git_hooks()
    setup_documentation()
    
    print("\n=== Setup Complete ===")
    print("""
Next steps:
1. Activate the virtual environment:
   - Windows: .\\venv\\Scripts\\activate
   - Linux/Mac: source venv/bin/activate
2. Review the documentation in docs/development-tools.md
3. Start developing!
""")

if __name__ == "__main__":
    main() 