from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from bcd_lab.data.preprocessors import (
    get_preprocessor,
    list_preprocessors,
    preprocess_dataset,
)
from bcd_lab.experiments import load_sweep_config, run_sweep

PROJECT_ROOT = Path.cwd().resolve()


def _config_entries() -> dict[str, Path]:
    """Return canonical config keys mapped to files below ``configs/``."""
    entries: dict[str, Path] = {}
    directory = PROJECT_ROOT / "configs"
    if not directory.exists():
        return entries
    for pattern in ("*.yaml", "*.yml"):
        for path in sorted(directory.rglob(pattern)):
            key = path.relative_to(directory).with_suffix("").as_posix()
            if key in entries:
                raise ValueError(
                    f"Duplicate experiment config key {key!r}: {entries[key]} and {path}"
                )
            entries[key] = path.resolve()
    return entries


def _config_keys() -> list[str]:
    return sorted(_config_entries())


def _resolve_config_path(value: str, *, by_name: bool = False) -> Path | None:
    if by_name:
        name = value
        if name.endswith(".yml") or name.endswith(".yaml"):
            name = str(Path(name).with_suffix(""))
        key = Path(name).as_posix().strip("/")
        if not key or ".." in Path(key).parts:
            return None
        return _config_entries().get(key)

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists() and path.is_file():
        return path.resolve()

    return _resolve_config_path(value, by_name=True)


def _resolve_output_dir(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _plot_results(config_path: Path, results_dir: Path | None = None) -> None:
    from bcd_lab.plotting import run_plots

    run_plots(PROJECT_ROOT, yaml_name=str(config_path), results_dir=results_dir)


def _handle_run_direct(
    config_name: str,
    output_dir_str: str | None = None,
    *,
    plot_after: bool = False,
    overwrite: bool = False,
) -> bool:
    config_path = _resolve_config_path(config_name)
    if config_path is None or not config_path.exists():
        print(f"Config not found: {config_name}")
        return False

    output_dir = _resolve_output_dir(output_dir_str)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"

    print(f"Running sweep configuration: {config_path}")
    run_sweep(config_path, output_root=output_dir, overwrite=overwrite)
    if plot_after:
        experiment = load_sweep_config(config_path).experiment
        _plot_results(config_path, output_dir / experiment)
    return True


def _handle_plot_direct(config_name: str, results_dir_str: str | None = None) -> bool:
    config_path = _resolve_config_path(config_name)
    if config_path is None or not config_path.exists():
        print(f"Config for plotting not found: {config_name}")
        return False

    results_dir = _resolve_output_dir(results_dir_str)
    if results_dir is None:
        experiment = load_sweep_config(config_path).experiment
        results_dir = PROJECT_ROOT / "results" / experiment
    _plot_results(config_path, results_dir)
    return True


def _handle_list() -> None:
    configs = _config_keys()
    if not configs:
        print("No experiment configs found.")
        return
    print("Available experiment configs:")
    for name in configs:
        print(f"  {name}")


def _handle_preprocess(
    dataset: str | None,
    *,
    list_only: bool,
    raw_dir: str,
    out_dir: str,
) -> bool:
    if list_only:
        print("Registered preprocessors:")
        for name in list_preprocessors():
            print(f"  {name:<20}  {get_preprocessor(name).description}")
        return True
    if dataset is None:
        print("preprocess requires a dataset name or --list", file=sys.stderr)
        return False
    raw_path = Path(raw_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    print(f"Preprocessing {dataset!r}")
    try:
        result = preprocess_dataset(dataset, raw_path, out_path)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False
    print(f"Clean data written to: {result}")
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and plot bcd-lab experiment configurations.",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE",
        help="import a registration module before dispatch (repeatable)",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run an experiment")
    run_parser.add_argument("config", help="config path or name from configs/")
    run_parser.add_argument("output_dir", nargs="?", help="results root (default: results/)")
    run_parser.add_argument("--plot", action="store_true", help="plot after a successful run")
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing artifacts for the planned run IDs",
    )

    plot_parser = subparsers.add_parser("plot", help="plot existing experiment results")
    plot_parser.add_argument("config", help="config path or name from configs/")
    plot_parser.add_argument(
        "results_dir", nargs="?", help="directory containing result JSON files"
    )

    subparsers.add_parser("list", help="list configs available under configs/")
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="convert a supported raw dataset to canonical CSV"
    )
    _configure_preprocess_parser(preprocess_parser)
    return parser


def _configure_preprocess_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", nargs="?", help="registered preprocessor name")
    parser.add_argument("--list", dest="list_only", action="store_true")
    parser.add_argument("--raw-dir", default="./data-raw", metavar="DIR")
    parser.add_argument("--out-dir", default="./data", metavar="DIR")


def _load_plugins(module_names: list[str]) -> None:
    for module_name in module_names:
        importlib.import_module(module_name)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _build_parser().print_help()
        return 0

    if args[0] in {"-h", "--help"}:
        _build_parser().print_help()
        return 0

    parsed = _build_parser().parse_args(args)
    _load_plugins(parsed.plugin)
    if parsed.command == "run":
        return (
            0
            if _handle_run_direct(
                parsed.config,
                parsed.output_dir,
                plot_after=parsed.plot,
                overwrite=parsed.overwrite,
            )
            else 1
        )
    if parsed.command == "plot":
        return 0 if _handle_plot_direct(parsed.config, parsed.results_dir) else 1
    if parsed.command == "list":
        _handle_list()
        return 0
    if parsed.command == "preprocess":
        return (
            0
            if _handle_preprocess(
                parsed.dataset,
                list_only=parsed.list_only,
                raw_dir=parsed.raw_dir,
                out_dir=parsed.out_dir,
            )
            else 1
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
