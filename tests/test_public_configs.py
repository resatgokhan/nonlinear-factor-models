"""Acceptance tests for the public example configurations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from bcd_lab.checkpoints import load_checkpoint_archive
from bcd_lab.experiments.composition import (
    expand_experiment_cases,
    preflight_experiment_cases,
)
from bcd_lab.experiments.loader import load_sweep_config
from bcd_lab.experiments.runner import run_sweep
from bcd_lab.plotting import run_plots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_public_configs_are_canonical_and_preflight() -> None:
    expected_case_counts = {
        "examples/quickstart.yaml": 1,
        "examples/synthetic_link_recovery.yaml": 1,
        "figures/figure_01_sigmoid_noise.yaml": 9,
        "figures/figure_02_joint_recovery_noise.yaml": 7,
        "figures/figure_03_identity_sample_size.yaml": 8,
        "figures/figure_04_joint_regularization.yaml": 9,
        "real_data/jester_learned_link.yaml": 1,
        "real_data/movielens_learned_link.yaml": 1,
        "studies/movielens_adam_learning_rate.yaml": 4,
        "studies/movielens_optimizer_comparison.yaml": 2,
        "studies/z_learning_rate.yaml": 3,
    }
    tracked_names = {path.relative_to(CONFIG_DIR).as_posix() for path in CONFIG_DIR.rglob("*.yaml")}
    assert tracked_names == set(expected_case_counts)

    for name, expected_count in expected_case_counts.items():
        spec = load_sweep_config(CONFIG_DIR / name)
        cases = expand_experiment_cases(spec)
        assert len(cases) == expected_count
        preflight_experiment_cases(cases)

        if spec.link["kind"] == "identity":
            assert spec.link["params"] == {}
        if spec.initialization["mode"] == "separate":
            assert set(spec.initialization) == {"mode", "z", "phi"}

    movielens = load_sweep_config(CONFIG_DIR / "real_data/movielens_learned_link.yaml")
    assert "type" not in movielens.data
    assert "noise_std" not in movielens.data
    assert movielens.link["kind"] == "piecewise_linear_rkhs"
    assert movielens.link["params"]["grid"] == {
        "count": 40,
        "bounds": {
            "name": "input_quantiles",
            "params": {
                "lower_quantile": 0.01,
                "upper_quantile": 0.99,
                "pad_fraction": 0.25,
                "min_padding": 0.5,
            },
        },
        "rebuild": {"name": "never", "params": {}},
    }
    assert movielens.algorithm.params["z_optimizer"] == {
        "name": "gradient_descent",
        "params": {"learning_rate": 8.0e-6},
    }
    assert movielens.algorithm.params["phi_update"] == {"learning_rate": 2.0e-6}
    assert movielens.algorithm.params["regularization"]["alpha"] == 1.0e-3
    assert movielens.algorithm.params["regularization"]["mu"] == 0.5
    assert movielens.algorithm.params["project_z"] is True
    assert movielens.runtime["max_iter"] == 1600
    assert movielens.selection is not None
    assert movielens.selection.metric == "validation_rmse"
    assert movielens.selection.mode == "min"
    assert movielens.selection.retain_checkpoint is True
    assert movielens.plot["plot3"]["type"] == "selected_link"

    sweep = load_sweep_config(CONFIG_DIR / "studies/z_learning_rate.yaml")
    assert set(sweep.diagnostics["families"]) == {
        "objective",
        "reconstruction",
        "truth_matrix",
    }

    sample_size = load_sweep_config(CONFIG_DIR / "figures/figure_03_identity_sample_size.yaml")
    assert sample_size.data["shape"] == {
        "n_rows": 1000,
        "n_cols": 1000,
        "rank": 10,
        "n_observations": 10000,
    }
    assert sample_size.runtime["max_iter"] == 10000
    assert sample_size.data["desirable"] is False
    assert sample_size.data["generating_link"]["name"] == "identity"
    assert sample_size.sweep.axes[0].values == [
        10000,
        20000,
        30000,
        40000,
        50000,
        60000,
        70000,
        80000,
    ]

    joint_noise = load_sweep_config(CONFIG_DIR / "figures/figure_02_joint_recovery_noise.yaml")
    assert joint_noise.data["shape"] == {
        "n_rows": 100,
        "n_cols": 100,
        "rank": 3,
        "n_observations": 5000,
    }
    assert joint_noise.algorithm.params["regularization"]["lambda_"] == 0.5
    assert joint_noise.algorithm.params["regularization"]["alpha"] == 1.0e-3
    assert joint_noise.algorithm.params["z_optimizer"]["params"]["learning_rate"] == 1.0e-5
    assert joint_noise.algorithm.params["phi_update"]["learning_rate"] == 1.0e-4
    assert joint_noise.runtime["max_iter"] == 20000
    assert joint_noise.data["generating_link"]["name"] == "random_monotone_rkhs"
    assert joint_noise.data["evaluation_domain"] == {
        "central_mass": 0.95,
        "grid_count": 1000,
    }
    assert joint_noise.link["params"]["grid"]["count"] == 1000
    assert joint_noise.diagnostics["families"]["phi_sharp"] == [
        "regret",
        "phi_sharp_diff",
        "phi_sharp_diff_star",
        "lyapunov",
    ]
    assert joint_noise.sweep.axes[0].values == [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    regularization = load_sweep_config(CONFIG_DIR / "figures/figure_04_joint_regularization.yaml")
    assert regularization.data["noise_std"] == 0.1
    assert regularization.sweep.axes[0].values == [
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    ]


def test_validation_selection_exports_one_coherent_checkpoint(tmp_path: Path) -> None:
    csv_path = tmp_path / "ratings.csv"
    rows = ["row,col,value"]
    rows.extend(f"{row},{column},{1.0 + row + column}" for row in range(4) for column in range(4))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    raw = yaml.safe_load(
        (CONFIG_DIR / "real_data/movielens_learned_link.yaml").read_text(encoding="utf-8")
    )
    raw["experiment"] = "selected_checkpoint"
    raw["data"]["path"] = str(csv_path)
    raw["data"]["shape"]["rank"] = 2
    raw["data"]["split"]["validation_fraction"] = 0.25
    raw["runtime"].update(
        {
            "max_iter": 3,
            "tol": 0.0,
            "diagnostics_interval": 1,
            "save_checkpoints": False,
            "checkpoint_iters": [],
        }
    )
    raw["plot"] = {
        "enabled": True,
        "name": "selected_checkpoint",
        "show_legend": False,
        "plot1": {
            "type": "selected_link",
            "show_identity": True,
            "pdf": "selected_checkpoint",
        },
    }
    config_path = tmp_path / "selection.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    run_sweep(config_path, output_root=tmp_path / "results")

    results_dir = tmp_path / "results" / "selected_checkpoint"
    artifact_path = next(results_dir.glob("*.json"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    selection = artifact["selection"]
    assert selection["metric"] == "validation_rmse"
    assert selection["diagnostics"] in artifact["diagnostics"]
    assert selection["diagnostics"]["iteration"] == selection["iteration"]
    assert selection["diagnostics"]["validation_rmse"] == selection["value"]
    assert selection["exports"]["link_function"]["kind"] == "piecewise_linear_rkhs"
    assert set(selection["exports"]["link_curve"]) == {"inputs", "outputs"}
    assert set(selection["exports"]["inner_matrix_range"]) == {"min", "max"}

    checkpoint_path = results_dir / selection["checkpoint_file"]
    checkpoints = load_checkpoint_archive(checkpoint_path)
    assert set(checkpoints) == {selection["iteration"]}
    assert not list(results_dir.glob("*_checkpoints.npz"))

    plot_dir = tmp_path / "plots"
    run_plots(
        PROJECT_ROOT,
        yaml_name=str(config_path),
        results_dir=results_dir,
        plots_dir=plot_dir,
    )
    assert list(plot_dir.glob("*.pdf"))


def test_quickstart_is_a_finite_improving_real_run(tmp_path: Path) -> None:
    run_sweep(CONFIG_DIR / "examples/quickstart.yaml", output_root=tmp_path)

    artifacts = list((tmp_path / "quickstart").glob("*.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text())
    diagnostics = payload["diagnostics"]

    for metric in ("loss", "x_error", "train_rmse"):
        values = [float(record[metric]) for record in diagnostics]
        assert all(math.isfinite(value) for value in values)
        assert values[-1] < values[0]
