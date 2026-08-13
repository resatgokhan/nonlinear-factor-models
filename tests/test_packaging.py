"""Tests for resources required by non-editable package installations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILENAMES = {
    "bcd.yaml",
    "data.yaml",
    "diagnostics.yaml",
    "initialization.yaml",
    "link.yaml",
    "runtime.yaml",
}


def test_wheel_contains_and_loads_default_configs(tmp_path: Path) -> None:
    """Packaged defaults remain readable when the package lives inside a wheel."""
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
            "--wheel-dir",
            str(wheel_dir),
            str(PROJECT_ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = list(wheel_dir.glob("bcd_lab-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    resource_prefix = "bcd_lab/experiments/default_configs/"
    with ZipFile(wheel) as archive:
        packaged_defaults = {
            Path(name).name
            for name in archive.namelist()
            if name.startswith(resource_prefix) and name.endswith(".yaml")
        }
        entry_points = archive.read(
            next(
                name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
            )
        ).decode("utf-8")
    assert packaged_defaults == DEFAULT_CONFIG_FILENAMES
    assert "bcd-lab = bcd_lab.cli:main" in entry_points

    minimal_config = tmp_path / "minimal.yaml"
    minimal_config.write_text(
        "schema_version: 2\n"
        "experiment: packaged-defaults\n"
        "algorithm: {name: bcd, params: {}}\n"
        "objective: {name: squared_error, params: {}}\n"
        "seed: 0\n",
        encoding="utf-8",
    )
    check_script = """
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

from bcd_lab.experiments import loader
from bcd_lab.checkpoints import CHECKPOINT_FORMAT_VERSION
from bcd_lab.data import list_generating_links
from bcd_lab.data.preprocessors import list_preprocessors

assert not Path(loader.__file__).is_file()
assert CHECKPOINT_FORMAT_VERSION == 1
assert loader.load_default_block("bcd")["phi_update"]["learning_rate"] == 0.001
assert loader.load_default_block("bcd")["z_optimizer"]["name"] == "gradient_descent"
config = loader.load_sweep_config(sys.argv[2])
assert config.data["source"] == "synthetic"
assert config.data["generating_link"]["name"] == "identity"
assert config.runtime["max_iter"] == 1000
assert config.algorithm.params["regularization"]["mu"] == 1.0
assert "ml-100k" in list_preprocessors()
assert "random_monotone_rkhs" in list_generating_links()
"""
    subprocess.run(
        [sys.executable, "-c", check_script, str(wheel), str(minimal_config)],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
