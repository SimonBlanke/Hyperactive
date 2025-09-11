"""
Test script for all Hyperactive v5 examples.

This test script runs all examples in the examples directory and verifies
that they execute without errors. It's designed for continuous integration
and regression testing to ensure all examples remain functional.

The test captures stdout/stderr but focuses on successful execution rather
than output validation, making it suitable for verifying that API changes
don't break the examples.
"""

import sys
import subprocess
from pathlib import Path
import pytest


def get_hyperactive_root():
    """Get the root directory of the Hyperactive project."""
    # Since test file is now in examples/, go up one level to find the root
    current = Path(__file__).parent.parent
    return current


def find_all_python_examples():
    """Find all Python example files in the examples directory."""
    root = get_hyperactive_root()
    examples_dir = root / "examples"

    if not examples_dir.exists():
        return []

    python_files = []

    # Find all .py files recursively in examples directory
    for py_file in examples_dir.rglob("*.py"):
        # Skip __pycache__, test files, and other non-example files
        if ("__pycache__" not in str(py_file) and
            ".pytest_cache" not in str(py_file) and
            not py_file.name.startswith("test_")):
            python_files.append(py_file)

    return sorted(python_files)


# Generate the list of examples at module level for pytest parametrize
ALL_EXAMPLES = find_all_python_examples()


def run_example(example_path, timeout=120):
    """
    Run a single example file and return success status and output.

    Parameters
    ----------
    example_path : Path
        Path to the example Python file
    timeout : int, default=120
        Maximum time to wait for example to complete (seconds)

    Returns
    -------
    success : bool
        True if example ran without errors
    stdout : str
        Standard output from the example
    stderr : str
        Standard error from the example
    """
    try:
        # Run the example with timeout
        result = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=example_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        success = result.returncode == 0
        return success, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", f"Example timed out after {timeout} seconds"
    except Exception as e:
        return False, "", f"Error running example: {str(e)}"


class TestExamples:
    """Test class for all Hyperactive examples."""

    @pytest.mark.parametrize("example_path", ALL_EXAMPLES, ids=lambda x: str(x.relative_to(get_hyperactive_root())) if ALL_EXAMPLES else "no_examples")
    def test_example_runs_successfully(self, example_path):
        """Test that an example runs without errors."""
        if not ALL_EXAMPLES:
            pytest.skip("No examples found")

        # Get relative path for cleaner test names
        root = get_hyperactive_root()
        relative_path = example_path.relative_to(root)

        print(f"\\nRunning example: {relative_path}")

        # Run the example
        success, stdout, stderr = run_example(example_path)

        # Print output for debugging if needed
        if stdout and len(stdout.strip()) > 0:
            print(f"STDOUT:\\n{stdout[:500]}{'...' if len(stdout) > 500 else ''}")

        if stderr and len(stderr.strip()) > 0:
            print(f"STDERR:\\n{stderr[:500]}{'...' if len(stderr) > 500 else ''}")

        # Assert that the example ran successfully
        assert success, f"Example failed: {relative_path}\\nSTDERR: {stderr}"

    def test_examples_directory_exists(self):
        """Test that the examples directory exists and contains files."""
        root = get_hyperactive_root()
        examples_dir = root / "examples"

        assert examples_dir.exists(), f"Examples directory not found: {examples_dir}"
        assert examples_dir.is_dir(), f"Examples path is not a directory: {examples_dir}"

        # Check that we have some examples
        python_files = list(examples_dir.rglob("*.py"))
        assert len(python_files) > 0, "No Python example files found"

        print(f"Found {len(python_files)} Python example files")

    def test_backend_directories_exist(self):
        """Test that all expected backend directories exist."""
        root = get_hyperactive_root()
        examples_dir = root / "examples"

        expected_backends = ["gfo", "sklearn", "optuna"]

        for backend in expected_backends:
            backend_dir = examples_dir / backend
            assert backend_dir.exists(), f"Backend directory not found: {backend_dir}"
            assert backend_dir.is_dir(), f"Backend path is not a directory: {backend_dir}"

            # Check that each backend has Python files
            py_files = list(backend_dir.glob("*.py"))
            assert len(py_files) > 0, f"No Python files found in {backend} backend"

    def test_readme_files_exist(self):
        """Test that README files exist for each backend."""
        root = get_hyperactive_root()
        examples_dir = root / "examples"

        expected_backends = ["gfo", "sklearn", "optuna"]

        for backend in expected_backends:
            readme_path = examples_dir / backend / "README.md"
            assert readme_path.exists(), f"README.md not found for {backend} backend"
            assert readme_path.is_file(), f"README.md is not a file for {backend} backend"

            # Check that README is not empty
            content = readme_path.read_text()
            assert len(content.strip()) > 0, f"README.md is empty for {backend} backend"


def main():
    """Main function to run tests directly."""
    # Find all examples
    examples = find_all_python_examples()

    if not examples:
        print("No example files found!")
        return 1

    print(f"Found {len(examples)} example files to test")

    failed_examples = []

    # Test each example
    for i, example_path in enumerate(examples, 1):
        root = get_hyperactive_root()
        relative_path = example_path.relative_to(root)

        print(f"[{i}/{len(examples)}] Testing: {relative_path}")

        success, stdout, stderr = run_example(example_path)

        if success:
            print(f"✓ PASSED: {relative_path}")
        else:
            print(f"✗ FAILED: {relative_path}")
            if stderr:
                print(f"  Error: {stderr[:200]}{'...' if len(stderr) > 200 else ''}")
            failed_examples.append(relative_path)

    # Print summary
    print(f"\\n{'='*50}")
    print(f"SUMMARY: {len(examples) - len(failed_examples)}/{len(examples)} examples passed")

    if failed_examples:
        print(f"\\nFailed examples:")
        for failed in failed_examples:
            print(f"  - {failed}")
        return 1
    else:
        print("\\n🎉 All examples passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
