"""Focused tests for the experiment launcher command contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bcd_lab import cli


def test_list_command_prints_available_configs(capsys) -> None:
    assert cli.main(["list"]) == 0

    output = capsys.readouterr().out
    assert "Available experiment configs:" in output
    assert "examples/quickstart" in output


def test_run_command_resolves_name_without_running_a_sweep() -> None:
    with patch.object(cli, "run_sweep") as run_sweep:
        assert cli.main(["run", "examples/quickstart"]) == 0

    config_path = cli.PROJECT_ROOT / "configs" / "examples" / "quickstart.yaml"
    run_sweep.assert_called_once_with(
        config_path.resolve(),
        output_root=cli.PROJECT_ROOT / "results",
        overwrite=False,
    )


def test_run_plot_flag_uses_the_experiment_results_directory() -> None:
    output_root = cli.PROJECT_ROOT / "custom-results"
    with (
        patch.object(cli, "run_sweep") as run_sweep,
        patch.object(cli, "load_sweep_config", return_value=SimpleNamespace(experiment="demo")),
        patch.object(cli, "_plot_results") as plot_results,
    ):
        assert cli.main(["run", "examples/quickstart", "custom-results", "--plot"]) == 0

    config_path = (cli.PROJECT_ROOT / "configs" / "examples" / "quickstart.yaml").resolve()
    run_sweep.assert_called_once_with(config_path, output_root=output_root, overwrite=False)
    plot_results.assert_called_once_with(config_path, output_root / "demo")


def test_run_overwrite_is_explicit_and_plugins_load_before_dispatch() -> None:
    with (
        patch.object(cli, "run_sweep") as run_sweep,
        patch.object(cli.importlib, "import_module") as import_module,
    ):
        assert (
            cli.main(
                [
                    "--plugin",
                    "research.extensions",
                    "run",
                    "examples/quickstart",
                    "--overwrite",
                ]
            )
            == 0
        )

    import_module.assert_called_once_with("research.extensions")
    config_path = (cli.PROJECT_ROOT / "configs" / "examples" / "quickstart.yaml").resolve()
    run_sweep.assert_called_once_with(
        config_path,
        output_root=cli.PROJECT_ROOT / "results",
        overwrite=True,
    )


def test_plot_command_does_not_run_an_experiment() -> None:
    with (
        patch.object(cli, "run_sweep") as run_sweep,
        patch.object(cli, "_plot_results") as plot_results,
    ):
        assert cli.main(["plot", "examples/quickstart", "saved-results"]) == 0

    config_path = (cli.PROJECT_ROOT / "configs" / "examples" / "quickstart.yaml").resolve()
    run_sweep.assert_not_called()
    plot_results.assert_called_once_with(config_path, cli.PROJECT_ROOT / "saved-results")


def test_plot_command_uses_configured_experiment_name_by_default() -> None:
    with (
        patch.object(cli, "load_sweep_config", return_value=SimpleNamespace(experiment="demo")),
        patch.object(cli, "_plot_results") as plot_results,
    ):
        assert cli.main(["plot", "examples/quickstart"]) == 0

    config_path = (cli.PROJECT_ROOT / "configs" / "examples" / "quickstart.yaml").resolve()
    plot_results.assert_called_once_with(config_path, cli.PROJECT_ROOT / "results" / "demo")


def test_help_and_empty_invocation_print_the_canonical_parser(capsys) -> None:
    assert cli.main(["--help"]) == 0
    assert "{run,plot,list,preprocess}" in capsys.readouterr().out

    assert cli.main([]) == 0
    assert "{run,plot,list,preprocess}" in capsys.readouterr().out


def test_nested_config_discovery_is_canonical_and_duplicate_safe(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "configs" / "figures"
    config_dir.mkdir(parents=True)
    yaml_path = config_dir / "recovery.yaml"
    yaml_path.write_text("experiment: recovery\n", encoding="utf-8")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    assert cli._config_keys() == ["figures/recovery"]
    assert cli._resolve_config_path("figures/recovery") == yaml_path.resolve()
    assert cli._resolve_config_path("recovery") is None

    yaml_path.with_suffix(".yml").write_text("experiment: duplicate\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate experiment config key"):
        cli._config_entries()
