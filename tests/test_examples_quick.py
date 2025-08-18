"""
Quick test for representative Hyperactive v5 examples.

This is a faster version for CI/CD that tests a few examples from each
backend to ensure the basic functionality is working without running
all 31 examples (which would take too long for CI).
"""

import subprocess
import sys
from pathlib import Path
import pytest


def get_hyperactive_root():
    """Get the root directory of the Hyperactive project."""
    current = Path(__file__).parent
    while current.name != "Hyperactive" and current.parent != current:
        current = current.parent
    return current


def run_example(example_path, timeout=60):
    """Run a single example file and return success status and output."""
    try:
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


class TestQuickExamples:
    """Quick test class for representative examples from each backend."""
    
    def test_gfo_random_search(self):
        """Test GFO backend with random search example."""
        root = get_hyperactive_root()
        example_path = root / "examples" / "gfo" / "random_search_example.py"
        
        if not example_path.exists():
            pytest.skip("GFO random search example not found")
            
        success, stdout, stderr = run_example(example_path)
        assert success, f"GFO random search failed: {stderr}"
        assert "optimization completed successfully" in stdout.lower()
    
    def test_gfo_hill_climbing(self):
        """Test GFO backend with hill climbing example."""
        root = get_hyperactive_root()
        example_path = root / "examples" / "gfo" / "hill_climbing_example.py"
        
        if not example_path.exists():
            pytest.skip("GFO hill climbing example not found")
            
        success, stdout, stderr = run_example(example_path)
        assert success, f"GFO hill climbing failed: {stderr}"
        assert "optimization completed successfully" in stdout.lower()
    
    def test_sklearn_random_search(self):
        """Test sklearn backend with random search example."""
        root = get_hyperactive_root()
        example_path = root / "examples" / "sklearn" / "random_search_example.py"
        
        if not example_path.exists():
            pytest.skip("Sklearn random search example not found")
            
        success, stdout, stderr = run_example(example_path)
        assert success, f"Sklearn random search failed: {stderr}"
        assert "best score:" in stdout.lower()
    
    def test_sklearn_grid_search(self):
        """Test sklearn backend with grid search example."""
        root = get_hyperactive_root()
        example_path = root / "examples" / "sklearn" / "grid_search_example.py"
        
        if not example_path.exists():
            pytest.skip("Sklearn grid search example not found")
            
        success, stdout, stderr = run_example(example_path)
        assert success, f"Sklearn grid search failed: {stderr}"
        assert "best score:" in stdout.lower()
    
    def test_optuna_tpe_sampler(self):
        """Test Optuna backend with TPE sampler example."""
        root = get_hyperactive_root()
        example_path = root / "examples" / "optuna" / "tpe_sampler_example.py"
        
        if not example_path.exists():
            pytest.skip("Optuna TPE sampler example not found")
            
        success, stdout, stderr = run_example(example_path, timeout=90)
        assert success, f"Optuna TPE sampler failed: {stderr}"
        assert "best score:" in stdout.lower()
    
    def test_optuna_random_sampler(self):
        """Test Optuna backend with random sampler example."""
        root = get_hyperactive_root()
        example_path = root / "examples" / "optuna" / "random_sampler_example.py"
        
        if not example_path.exists():
            pytest.skip("Optuna random sampler example not found")
            
        success, stdout, stderr = run_example(example_path)
        assert success, f"Optuna random sampler failed: {stderr}"
        assert "best score:" in stdout.lower()
    
    def test_examples_structure(self):
        """Test that the expected directory structure exists."""
        root = get_hyperactive_root()
        examples_dir = root / "examples"
        
        assert examples_dir.exists(), "Examples directory not found"
        
        # Check backend directories exist
        for backend in ["gfo", "sklearn", "optuna"]:
            backend_dir = examples_dir / backend
            assert backend_dir.exists(), f"{backend} backend directory not found"
            assert backend_dir.is_dir(), f"{backend} is not a directory"
            
            # Check README exists
            readme = backend_dir / "README.md"
            assert readme.exists(), f"README.md not found in {backend} directory"
            
            # Check that there are Python files
            py_files = list(backend_dir.glob("*.py"))
            assert len(py_files) > 0, f"No Python files found in {backend} directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])