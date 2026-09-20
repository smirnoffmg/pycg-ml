"""Regression tests for running the analyser on Python 3.12+.

Upstream PyCG installs its stub loader as a global path hook and then calls
importlib.invalidate_caches(). Since Python 3.12 that call imports parts of the
standard library lazily, so the hook fires for modules the analyser itself needs
and the run dies — either on an edge with no current module, or on a stdlib
module served an empty source.
"""

import json
import subprocess
import sys
import textwrap

from pycg_ml.generator import CallGraphGenerator
from pycg_ml.utils.constants import CALL_GRAPH_OP


def write_package(root):
    (root / "helpers.py").write_text(
        textwrap.dedent("""
            def scale(x):
                return x * 2
        """)
    )
    entry = root / "main.py"
    entry.write_text(
        textwrap.dedent("""
            from helpers import scale

            def run():
                return scale(21)

            run()
        """)
    )
    return entry


def test_generator_resolves_intra_package_calls(tmp_path):
    entry = write_package(tmp_path)

    generator = CallGraphGenerator([str(entry)], str(tmp_path), -1, CALL_GRAPH_OP)
    generator.analyze()
    graph = generator.output()

    assert set(graph["main.run"]) == {"helpers.scale"}
    assert "helpers.scale" in graph


def test_stdlib_stays_importable_while_hooks_are_installed(tmp_path):
    """The stub loader must not shadow the standard library.

    Run in a subprocess: the failure it guards against corrupts sys.modules for
    the whole interpreter.
    """
    entry = write_package(tmp_path)
    output = tmp_path / "cg.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycg_ml",
            "--package",
            str(tmp_path),
            str(entry),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["main.run"] == ["helpers.scale"]
