"""Test configs."""

# todo 5.1: review if skipping is still necessary
# if not, remove below
import sys
import platform

cpython_138031_bug_present = (
    sys.version_info[:3] == (3, 13, 7) and platform.system() == "Windows"
)
# end remove

# list of str, names of estimators to exclude from testing
# WARNING: tests for these estimators will be skipped
EXCLUDE_ESTIMATORS = [
    "DummySkipped",
    "ClassName",  # exclude classes from extension templates
]

# dictionary of lists of str, names of tests to exclude from testing
# keys are class names of estimators, values are lists of test names to exclude
# WARNING: tests with these names will be skipped

# cpython bug #138031 causes random test failures on Windows with Python 3.13.7
# see https://github.com/SimonBlanke/Hyperactive/issues/169
# todo 5.1: review if this skipping is still necessary
# if no longer necessary, remove the "if" condition and leave the "else" part
if cpython_138031_bug_present:
    EXCLUDED_TESTS = {
        "GridSearchSK": ["test_opt_run"],
        "RandomSearchSK": ["test_opt_run"],
    }
else:
    EXCLUDED_TESTS = {}
