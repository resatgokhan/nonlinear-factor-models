"""Focused tests for the public fixed-coordinate optimizer boundary."""

from __future__ import annotations

import numpy as np
import pytest

from bcd_lab.optimizers import (
    OptimizerSpec,
    build_optimizer,
    restore_optimizer,
    serialize_optimizer,
)


def test_gradient_descent_matches_the_canonical_array_step() -> None:
    parameter = np.array([1.0, -2.0], dtype=np.float64)
    gradient = np.array([3.0, -4.0])
    optimizer = build_optimizer(
        OptimizerSpec("gradient_descent", {"learning_rate": 0.25}),
        parameter,
    )

    update = optimizer.compute_update(gradient)

    assert np.array_equal(update, 0.25 * gradient)
    assert np.array_equal(parameter - update, np.array([0.25, -1.0]))


def test_adam_first_two_updates_match_bias_corrected_definition() -> None:
    parameter = np.zeros(2, dtype=np.float64)
    optimizer = build_optimizer(
        OptimizerSpec(
            "adam",
            {
                "learning_rate": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1.0e-8,
            },
        ),
        parameter,
    )
    gradients = [np.array([1.0, -2.0]), np.array([3.0, -4.0])]
    first = np.zeros(2)
    second = np.zeros(2)

    for step, gradient in enumerate(gradients, start=1):
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * np.square(gradient)
        expected = (
            0.01 * (first / (1.0 - 0.9**step)) / (np.sqrt(second / (1.0 - 0.999**step)) + 1.0e-8)
        )
        assert np.allclose(optimizer.compute_update(gradient), expected)


def test_adam_checkpoint_restores_the_same_next_update() -> None:
    parameter = np.zeros((2, 1), dtype=np.float32)
    spec = OptimizerSpec("adam", {"learning_rate": 0.01})
    optimizer = build_optimizer(spec, parameter)
    optimizer.compute_update(np.array([[1.0], [-2.0]], dtype=np.float32))

    restored = restore_optimizer(
        serialize_optimizer(optimizer),
        spec=spec,
        parameter=parameter,
    )
    next_gradient = np.array([[0.5], [3.0]], dtype=np.float32)

    assert np.array_equal(
        restored.compute_update(next_gradient),
        optimizer.compute_update(next_gradient),
    )


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        (OptimizerSpec("adam", {"learning_rate": 0.0}), "positive"),
        (OptimizerSpec("adam", {"learning_rate": 0.1, "beta1": 1.0}), r"\[0, 1\)"),
        (OptimizerSpec("adam", {"learning_rate": 0.1, "extra": 2}), "unknown"),
        (OptimizerSpec("missing", {"learning_rate": 0.1}), "Unknown optimizer"),
    ],
)
def test_invalid_optimizer_specs_fail_contextually(spec: OptimizerSpec, error: str) -> None:
    with pytest.raises((KeyError, TypeError, ValueError), match=error):
        build_optimizer(spec, np.zeros(2))


def test_optimizer_rejects_mismatched_and_non_finite_gradients() -> None:
    optimizer = build_optimizer(
        OptimizerSpec("adam", {"learning_rate": 0.1}),
        np.zeros(2),
    )
    with pytest.raises(ValueError, match="shape"):
        optimizer.compute_update(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        optimizer.compute_update(np.array([np.inf, 0.0]))
