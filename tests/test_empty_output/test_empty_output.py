import os, sys, subprocess, pytest


if sys.platform.startswith("win"):
    pytest.skip("skip these tests for windows", allow_module_level=True)


here = os.path.dirname(os.path.abspath(__file__))

verbose_file = os.path.join(here, "verbose.py")
non_verbose_file = os.path.join(here, "non_verbose.py")


def _run_subprocess(script):
    output = []
    process = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    # Read output line by line
    while True:
        line = process.stdout.readline()
        if line:
            output.append(line)
        if not line and process.poll() is not None:
            break

    return "".join(output), process.stderr.read()


def test_empty_output():
    stdout_verb, stderr_verb = _run_subprocess(verbose_file)
    stdout_non_verb, stderr_non_verb = _run_subprocess(non_verbose_file)

    print("\n stdout_verb \n", stdout_verb, "\n")
    print("\n stderr_verb \n", stderr_verb, "\n")

    print("\n stdout_non_verb \n", stdout_non_verb, "\n")
    print("\n stderr_non_verb \n", stderr_non_verb, "\n")

    assert "Results:" in stdout_verb
    assert not stdout_non_verb
