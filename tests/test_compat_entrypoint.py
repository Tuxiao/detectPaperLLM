from __future__ import annotations

import os
import subprocess
import sys


def test_legacy_entrypoint_forwards_to_new_cli(repo_root, pythonpath):
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, (repo_root / "ddl_finetune.py").as_posix(), "--help"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "prepare-pairs" in result.stdout
    assert "build-ref" in result.stdout
