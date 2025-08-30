#!/usr/bin/env python3
"""
Development utility script for the vLLM Inference Service.
Provides common development tasks in a single command.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True, capture_output=False):
    """Run a shell command with error handling."""
    try:
        print(f"[*] {description}...")
    except UnicodeEncodeError:
        print(f"[*] {description}...")

    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)

        try:
            print(f"[✓] {description} completed successfully")
        except UnicodeEncodeError:
            print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        try:
            print(f"[✗] {description} failed: {e}")
        except UnicodeEncodeError:
            print(f"[ERROR] {description} failed: {e}")

        if capture_output and e.stdout:
            print("STDOUT:", e.stdout)
        if capture_output and e.stderr:
            print("STDERR:", e.stderr)
        return False


def setup_dev():
    """Set up development environment."""
    print("Setting up development environment...")

    commands = [
        ("python -m pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install -r requirements-dev.txt", "Installing dev dependencies"),
        ("pre-commit install", "Installing pre-commit hooks"),
    ]

    for cmd, desc in commands:
        if not run_command(cmd, desc, check=False):
            print("[ERROR] Development setup failed")
            return False

    print("[OK] Development environment ready!")
    return True


def lint():
    """Run linting tools."""
    print("Running code quality checks...")

    commands = [
        ("ruff check app/ tests/", "Running ruff linter"),
        ("ruff format --check app/ tests/", "Checking code formatting"),
        ("mypy app/", "Running type checker"),
        ("bandit -r app/", "Running security checks"),
    ]

    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc, check=False):
            success = False

    return success


def format_code():
    """Format code using ruff."""
    commands = [
        ("ruff format app/ tests/", "Formatting code with ruff"),
        ("ruff check --fix app/ tests/", "Fixing linting issues"),
        ("isort app/ tests/", "Sorting imports"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc, check=False)


def test(coverage=True, verbose=False):
    """Run tests."""
    cmd = "pytest tests/"

    if verbose:
        cmd += " -v"

    if coverage:
        cmd += " --cov=app --cov-report=term-missing --cov-report=html"

    return run_command(cmd, "Running tests")


def build_docker():
    """Build Docker image."""
    tag = "vllm-inference:dev"
    cmd = f"docker build -t {tag} ."

    if run_command(cmd, "Building Docker image"):
        print(f"[OK] Docker image built: {tag}")
        return True
    return False


def run_docker():
    """Run Docker container locally."""
    if not build_docker():
        return False

    cmd = """docker run --rm -it \
        --gpus all \
        -p 8000:8000 \
        -e MODEL_NAME=microsoft/phi-2 \
        -e DEBUG=true \
        vllm-inference:dev"""

    print("[*] Starting Docker container...")
    print("Service will be available at http://localhost:8000")
    print("Press Ctrl+C to stop")

    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("[*] Stopping container...")
        return True


def load_test(concurrency=10, requests=100):
    """Run load test."""
    cmd = f"""python scripts/load_test.py \
        --url http://localhost:8000/v1/generate \
        --concurrency {concurrency} \
        --requests {requests} \
        --prompt "Test prompt for load testing" \
        --max-tokens 50"""

    return run_command(cmd, f"Running load test ({requests} requests, {concurrency} concurrent)")


def clean():
    """Clean up build artifacts and cache."""
    commands = [
        ("find . -type d -name __pycache__ -exec rm -rf {} +", "Removing Python cache"),
        ("find . -type f -name '*.pyc' -delete", "Removing .pyc files"),
        ("rm -rf .pytest_cache/", "Removing pytest cache"),
        ("rm -rf htmlcov/", "Removing coverage HTML reports"),
        ("rm -rf .coverage", "Removing coverage data"),
        ("rm -rf dist/", "Removing distribution files"),
        ("rm -rf *.egg-info/", "Removing egg info"),
        ("docker system prune -f", "Cleaning up Docker"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc, check=False)


def security_scan():
    """Run comprehensive security scan."""
    commands = [
        ("safety check", "Checking dependencies for vulnerabilities"),
        ("bandit -r app/ -f json -o bandit-report.json", "Running security analysis"),
        (
            "docker run --rm -v $(pwd):/app -w /app aquasec/trivy fs .",
            "Running Trivy filesystem scan",
        ),
    ]

    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc, check=False):
            success = False

    return success


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="vLLM Inference Service Development Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    subparsers.add_parser("setup", help="Set up development environment")

    # Code quality commands
    subparsers.add_parser("lint", help="Run linting and type checking")
    subparsers.add_parser("format", help="Format code")

    # Testing commands
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--no-coverage", action="store_true", help="Skip coverage report")
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Docker commands
    subparsers.add_parser("build", help="Build Docker image")
    subparsers.add_parser("run", help="Run service in Docker")

    # Load testing
    load_parser = subparsers.add_parser("load-test", help="Run load test")
    load_parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrency level")
    load_parser.add_argument("-r", "--requests", type=int, default=100, help="Number of requests")

    # Utility commands
    subparsers.add_parser("clean", help="Clean up build artifacts")
    subparsers.add_parser("security", help="Run security scans")

    # CI command (run all checks)
    subparsers.add_parser("ci", help="Run all CI checks locally")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Change to project root
    project_root = Path(__file__).parent
    import os

    os.chdir(project_root)

    success = True

    if args.command == "setup":
        success = setup_dev()
    elif args.command == "lint":
        success = lint()
    elif args.command == "format":
        format_code()
    elif args.command == "test":
        success = test(coverage=not args.no_coverage, verbose=args.verbose)
    elif args.command == "build":
        success = build_docker()
    elif args.command == "run":
        success = run_docker()
    elif args.command == "load-test":
        success = load_test(args.concurrency, args.requests)
    elif args.command == "clean":
        clean()
    elif args.command == "security":
        success = security_scan()
    elif args.command == "ci":
        print("[*] Running full CI pipeline locally...")
        steps = [
            (lint, "Code quality checks"),
            (lambda: test(coverage=True), "Tests with coverage"),
            (build_docker, "Docker build"),
            (security_scan, "Security scans"),
        ]

        for step_func, step_name in steps:
            print(f"\n[*] {step_name}")
            if not step_func():
                success = False
                break

        if success:
            print("\n[OK] All CI checks passed!")
        else:
            print("\n[ERROR] CI pipeline failed")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
