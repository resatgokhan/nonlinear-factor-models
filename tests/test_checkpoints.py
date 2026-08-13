"""Round-trip coverage for component-owned checkpoint state."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest

from bcd_lab.algorithms import CheckpointableAlgorithm, get_algorithm
from bcd_lab.algorithms.state import init_state
from bcd_lab.checkpoints import load_checkpoint_archive, save_checkpoint_archive
from bcd_lab.config import (
    BcdConfig,
    LinkConfig,
    ModelConfig,
    PhiUpdateConfig,
    RegularizationConfig,
    RuntimeConfig,
    ShapeConfig,
)
from bcd_lab.diagnostics import DiagnosticRequest
from bcd_lab.experiments.runner import _save_run_results
from bcd_lab.function_models import register_function_model
from bcd_lab.losses import get_objective
from bcd_lab.optimizers import OptimizerSpec
from bcd_lab.types import DatasetBundle, SampleSet


class OffsetFunction:
    """Small external-style model whose state has no RKHS field names."""

    kind = "checkpoint_offset"
    checkpoint_version = 3

    def __init__(self, offset: float) -> None:
        self.offset = float(offset)

    def eval_array(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) + self.offset

    def copy(self) -> OffsetFunction:
        return OffsetFunction(self.offset)

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {"offset": self.offset}

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
    ) -> None:
        if version != self.checkpoint_version:
            raise ValueError(f"unsupported offset state version {version}")
        self.offset = float(state["offset"])


def _build_offset(config, params=None, dataset=None, z=None) -> OffsetFunction:
    _ = params, dataset, z
    return OffsetFunction(float(config.link.params.get("offset", 0.0)))


def _sample_set() -> SampleSet:
    return (
        np.array([1.0, 2.0, 3.0, 4.0]),
        (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])),
    )


def test_external_function_round_trips_through_algorithm_checkpoint(
    tmp_path,
) -> None:
    register_function_model(OffsetFunction.kind)(_build_offset)
    config = ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("gradient_descent", {"learning_rate": 0.1}),
        phi_update=PhiUpdateConfig(learning_rate=0.1),
        regularization=RegularizationConfig(lambda_=0.0, mu=0.0, alpha=0.0),
        link=LinkConfig(kind=OffsetFunction.kind, params={"offset": 0.0}),
        runtime=RuntimeConfig(
            max_iter=0,
            tol=0.0,
            dtype="float32",
            checkpoint_iters=(0,),
        ),
        bcd=BcdConfig(project_z=False, freeze_phi_updates=True),
    )
    dataset = DatasetBundle(
        source="checkpoint-test",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
    )
    z_initial = np.arange(4, dtype=np.float32).reshape(4, 1)
    algorithm = get_algorithm("bcd")
    assert isinstance(algorithm, CheckpointableAlgorithm)

    result = algorithm.run(
        config=config,
        dataset=dataset,
        objective=get_objective("squared_error"),
        z_init=z_initial,
        phi_init=OffsetFunction(2.75),
        diagnostics_request=DiagnosticRequest(families={"objective": {"loss"}}),
        rng=np.random.default_rng(0),
    )
    _save_run_results(
        tmp_path,
        "external-model",
        {"artifact_version": 1},
        result,
        save_checkpoints=True,
    )
    loaded = load_checkpoint_archive(tmp_path / "external-model_checkpoints.npz")
    checkpoint = loaded[0]
    restored = algorithm.restore_checkpoint(
        checkpoint,
        config=config,
        dataset=dataset,
    )

    assert checkpoint.format_version == 1
    assert checkpoint.state_version == algorithm.checkpoint_version
    assert checkpoint.payload["phi"]["state_version"] == 3
    assert np.array_equal(restored.z, z_initial)
    assert restored.z.dtype == z_initial.dtype
    assert isinstance(restored.phi, OffsetFunction)
    assert restored.phi.offset == 2.75
    assert restored.z_optimizer.name == "gradient_descent"
    assert np.array_equal(
        restored.phi.eval_array(np.array([-1.0, 1.0])),
        np.array([1.75, 3.75]),
    )


def test_checkpoint_archive_write_is_atomic_on_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoints.npz"
    target.write_bytes(b"original")

    def fail_save(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected archive failure")

    monkeypatch.setattr("bcd_lab.checkpoints.np.savez_compressed", fail_save)
    with pytest.raises(RuntimeError, match="injected archive failure"):
        save_checkpoint_archive(target, {})

    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_bcd_checkpoint_restores_adam_state_for_the_same_next_update() -> None:
    config = ModelConfig(
        shape=ShapeConfig(n_rows=2, n_cols=2, rank=1, n_observations=4),
        z_optimizer=OptimizerSpec("adam", {"learning_rate": 0.01}),
        phi_update=PhiUpdateConfig(learning_rate=0.0),
        regularization=RegularizationConfig(lambda_=0.0, mu=0.0, alpha=0.0),
        link=LinkConfig(kind="identity"),
        runtime=RuntimeConfig(max_iter=1, tol=0.0),
        bcd=BcdConfig(project_z=False, freeze_phi_updates=True),
    )
    dataset = DatasetBundle(
        source="checkpoint-test",
        n_rows=2,
        n_cols=2,
        rank=1,
        train=_sample_set(),
    )
    state = init_state(
        dataset.train,
        config,
        z_init=np.arange(4, dtype=np.float64).reshape(4, 1),
        phi_init=None,
    )
    state.z_optimizer.compute_update(np.ones_like(state.z))
    algorithm = get_algorithm("bcd")
    assert isinstance(algorithm, CheckpointableAlgorithm)

    checkpoint = algorithm.capture_checkpoint(state, elapsed_seconds=1.5)
    restored = algorithm.restore_checkpoint(checkpoint, config=config, dataset=dataset)
    next_gradient = np.array([[0.5], [-1.0], [2.0], [-3.0]])

    assert checkpoint.state_version == 2
    assert checkpoint.payload["z_optimizer"]["name"] == "adam"
    assert np.array_equal(
        restored.z_optimizer.compute_update(next_gradient),
        state.z_optimizer.compute_update(next_gradient),
    )
