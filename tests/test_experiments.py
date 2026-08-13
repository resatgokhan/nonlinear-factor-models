"""Unit tests for the experiment sweep runner package."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bcd_lab.experiments.loader import deep_merge, load_sweep_config
from bcd_lab.experiments.schema import SweepAxis
from bcd_lab.experiments.sweep import expand_axes, replace_nested


@dataclass
class DummyChildDC:
    val: int


@dataclass
class DummyParentDC:
    child: DummyChildDC
    params: dict[str, Any]


class TestExperiments(unittest.TestCase):
    def test_deep_merge(self) -> None:
        base = {
            "a": 1,
            "b": {"x": 10, "y": 20},
            "c": 3,
        }
        overrides = {
            "b": {"y": 25, "z": 30},
            "c": 4,
            "d": 5,
        }
        expected = {
            "a": 1,
            "b": {"x": 10, "y": 25, "z": 30},
            "c": 4,
            "d": 5,
        }
        self.assertEqual(deep_merge(base, overrides), expected)

    def test_expand_axes(self) -> None:
        axes = [
            SweepAxis(path="algorithm.params.phi_update.learning_rate", values=[0.1, 0.2]),
            SweepAxis(path="link.params.unconstrained_ridge", values=[1e-8]),
        ]
        cases = expand_axes(axes)
        expected = [
            {
                "algorithm.params.phi_update.learning_rate": 0.1,
                "link.params.unconstrained_ridge": 1e-8,
            },
            {
                "algorithm.params.phi_update.learning_rate": 0.2,
                "link.params.unconstrained_ridge": 1e-8,
            },
        ]
        self.assertEqual(cases, expected)

    def test_replace_nested(self) -> None:
        parent_dc = DummyParentDC(child=DummyChildDC(5), params={"a": 1})

        # Replace nested dictionary key
        parent_dc_2 = replace_nested(parent_dc, "params.a", 10)
        self.assertEqual(parent_dc_2.params["a"], 10)

        # Replace nested dataclass attribute
        parent_dc_3 = replace_nested(parent_dc, "child.val", 10)
        self.assertEqual(parent_dc_3.child.val, 10)

    def test_load_sweep_config(self) -> None:
        yaml_content = """
schema_version: 2
experiment: test_exp
algorithm:
  name: bcd
  params:
    regularization:
      lambda_: 0.05
    phi_update:
      learning_rate: 0.15
trials: 2

objective:
  name: squared_error
  params: {}

seed: 10

data:
  shape:
    n_rows: 20
    n_cols: 20
    rank: 2
    n_observations: 100

link:
  kind: gaussian_rkhs
  params:
    unconstrained_ridge: 1.0e-5

runtime:
  max_iter: 50
  tol: 1.0e-5

sweep:
  axes:
    - path: algorithm.params.phi_update.learning_rate
      values: [0.1, 0.2]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text(yaml_content)

            cfg = load_sweep_config(config_file)
            self.assertEqual(cfg.schema_version, 2)
            self.assertEqual(cfg.experiment, "test_exp")
            self.assertEqual(cfg.trials, 2)
            self.assertEqual(cfg.seed, 10)
            self.assertEqual(cfg.data["shape"]["n_rows"], 20)
            self.assertEqual(cfg.data["shape"]["n_observations"], 100)
            self.assertEqual(cfg.algorithm.params["regularization"]["lambda_"], 0.05)
            # Checked merged defaults
            self.assertEqual(cfg.algorithm.params["regularization"]["mu"], 1.0)
            self.assertEqual(cfg.algorithm.params["phi_update"]["learning_rate"], 0.15)
            self.assertEqual(
                cfg.algorithm.params["z_optimizer"]["params"]["learning_rate"],
                0.1,
            )
            self.assertEqual(cfg.link["kind"], "gaussian_rkhs")
            self.assertEqual(cfg.link["params"]["unconstrained_ridge"], 1e-5)
            self.assertEqual(cfg.link["params"]["monotone_phi_enabled"], True)  # default
            self.assertEqual(cfg.runtime["max_iter"], 50)
            self.assertEqual(cfg.runtime["tol"], 1.0e-5)
            self.assertEqual(cfg.runtime["save_checkpoints"], False)  # default
            self.assertEqual(cfg.algorithm.params["project_z"], True)
            self.assertEqual(cfg.algorithm.params["freeze_phi_updates"], False)
            self.assertEqual(len(cfg.sweep.axes), 1)
            self.assertEqual(
                cfg.sweep.axes[0].path,
                "algorithm.params.phi_update.learning_rate",
            )

    def test_removed_top_level_fields_are_rejected(self) -> None:
        yaml_content = """
schema_version: 2
experiment: invalid
algorithm: {name: bcd, params: {}}
objective: {name: squared_error, params: {}}
solver: bcd
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text(yaml_content)
            with self.assertRaisesRegex(ValueError, "Unknown top-level.*solver"):
                load_sweep_config(config_file)

    def test_resolve_output_dir(self) -> None:
        from bcd_lab import cli

        self.assertIsNone(cli._resolve_output_dir(None))
        resolved = cli._resolve_output_dir("my_out")
        self.assertEqual(resolved, cli.PROJECT_ROOT / "my_out")

    def test_resolve_config_path(self) -> None:
        from bcd_lab import cli

        # 1. Resolve existing config by name
        resolved = cli._resolve_config_path("examples/quickstart")
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.stem, "quickstart")

        # 2. Resolve non-existent config returns None
        resolved_missing = cli._resolve_config_path("missing_config_name_xyz")
        self.assertIsNone(resolved_missing)
