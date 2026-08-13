"""Common runtime types and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np

if TYPE_CHECKING:
    from bcd_lab.function_models.base import ScalarFunction

SampleSet = tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class FunctionReferences:
    """Optional reference functions supplied by a dataset.

    ``truth`` is the generating/reference link function, while ``initial`` is
    a reference initialization used for comparisons. ``sample_count`` controls
    deterministic quadrature diagnostics when an analytic distance is not
    requested.
    """

    truth: ScalarFunction | None = None
    initial: ScalarFunction | None = None
    sample_count: int = 1000

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("function reference sample_count must be positive")


@dataclass(frozen=True)
class FunctionEvaluationDomain:
    """Dataset-owned evenly spaced grid for scalar-function errors.

    ``central_mass`` records the probability mass retained from the latent
    truth distribution when the grid was constructed. It is ``None`` for
    domains supplied by loaders through another documented policy.
    """

    grid: np.ndarray
    source: str
    central_mass: float | None = None

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid).copy()
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError(
                "function evaluation grid must be one-dimensional with at least two points"
            )
        if not np.issubdtype(grid.dtype, np.number) or np.issubdtype(
            grid.dtype, np.complexfloating
        ):
            raise ValueError("function evaluation grid must contain real numbers")
        if not np.issubdtype(grid.dtype, np.floating):
            grid = grid.astype(float)
        if not np.all(np.isfinite(grid)):
            raise ValueError("function evaluation grid must be finite")
        spacing = np.diff(grid)
        if not np.all(spacing > 0.0):
            raise ValueError("function evaluation grid must be strictly increasing")
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("function evaluation grid must be evenly spaced")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("function evaluation domain source must be non-empty")
        if self.central_mass is not None:
            if (
                isinstance(self.central_mass, bool)
                or not isinstance(self.central_mass, Real)
                or not np.isfinite(self.central_mass)
                or not 0.0 < self.central_mass <= 1.0
            ):
                raise ValueError("function evaluation central_mass must be in (0, 1]")
        grid.setflags(write=False)
        object.__setattr__(self, "grid", grid)


@dataclass(frozen=True)
class SplitStatistics:
    """Stable counts and sampling details for a dataset's splits."""

    train_count: int
    validation_count: int = 0
    test_count: int = 0
    total_count: int | None = None
    validation_fraction: float | None = None
    seed: int | None = None
    stratify_by: str | None = None
    row_count_summary: Mapping[str, float] | None = None
    column_count_summary: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        counts = (self.train_count, self.validation_count, self.test_count)
        if any(count < 0 for count in counts):
            raise ValueError("split counts must be non-negative")
        if self.total_count is not None and self.total_count < sum(counts):
            raise ValueError("total_count cannot be smaller than the split counts")

    @property
    def resolved_total_count(self) -> int:
        """Return an explicit total or the sum of the represented splits."""
        if self.total_count is not None:
            return self.total_count
        return self.train_count + self.validation_count + self.test_count


@dataclass(frozen=True)
class IdMappings:
    """Mappings from source row/column identifiers to dense internal indices."""

    row_to_index: Mapping[object, int]
    column_to_index: Mapping[object, int]


@dataclass(frozen=True)
class SourceFingerprint:
    """Content identity for a mutable external dataset source."""

    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError(
                "source fingerprint sha256 must be 64 lowercase hexadecimal characters"
            )
        if self.size_bytes < 0:
            raise ValueError("source fingerprint size_bytes must be non-negative")


@dataclass(frozen=True)
class PreprocessingSource:
    """Content identity of one raw file consumed by a preprocessor."""

    source_id: str
    fingerprint: SourceFingerprint

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("preprocessing source_id must be non-empty")


@dataclass(frozen=True)
class IdentifierSummary:
    """Compact identity of the source row or column identifier set."""

    source_field: str
    count: int
    minimum: str
    maximum: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.source_field:
            raise ValueError("identifier source_field must be non-empty")
        if self.count <= 0:
            raise ValueError("identifier count must be positive")
        SourceFingerprint(self.sha256, 0)


@dataclass(frozen=True)
class PreprocessingProvenance:
    """Versioned record connecting a clean CSV to its raw inputs and policy."""

    schema_version: int
    preprocessor: str
    preprocessor_version: int
    parameters: Mapping[str, Any]
    sources: tuple[PreprocessingSource, ...]
    output_name: str
    output_fingerprint: SourceFingerprint
    counts: Mapping[str, int]
    identifiers: Mapping[str, IdentifierSummary]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported preprocessing provenance schema version")
        if not self.preprocessor or self.preprocessor_version <= 0:
            raise ValueError("preprocessor name and version must be valid")
        if not self.sources:
            raise ValueError("preprocessing provenance requires at least one raw source")
        if not self.output_name:
            raise ValueError("preprocessing output_name must be non-empty")
        if any(value < 0 for value in self.counts.values()):
            raise ValueError("preprocessing counts must be non-negative")


@dataclass(frozen=True)
class DatasetCapabilities:
    """Typed, optional dataset features consumed across core components.

    Loader-specific provenance and statistics that have no cross-component
    meaning belong in :attr:`DatasetBundle.metadata`, not here.
    """

    functions: FunctionReferences = field(default_factory=FunctionReferences)
    function_domain: FunctionEvaluationDomain | None = None
    linked_truth: np.ndarray | None = None
    splits: SplitStatistics | None = None
    id_mappings: IdMappings | None = None
    source_fingerprint: SourceFingerprint | None = None
    preprocessing: PreprocessingProvenance | None = None


@dataclass
class DatasetBundle:
    """Common dataset object passed to initializers, algorithms, and evaluators.

    Stable optional cross-component fields live in ``capabilities``. ``metadata``
    is intentionally reserved for loader-specific extras.
    """

    source: str
    n_rows: int
    n_cols: int
    rank: int
    train: SampleSet
    validation: SampleSet | None = None
    test: SampleSet | None = None
    dense_truth: np.ndarray | None = None
    latent_truth: np.ndarray | None = None
    capabilities: DatasetCapabilities = field(default_factory=DatasetCapabilities)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_observations(self) -> int:
        return int(np.asarray(self.train[0]).size)

    def validate(self) -> None:
        """Validate stable structure shared by every dataset loader."""
        for name, value in (("n_rows", self.n_rows), ("n_cols", self.n_cols), ("rank", self.rank)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"dataset {name} must be a positive integer")
        if self.rank > min(self.n_rows, self.n_cols):
            raise ValueError("dataset rank cannot exceed its row or column dimension")

        split_sizes: dict[str, int] = {}
        for split_name, samples in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if samples is None:
                split_sizes[split_name] = 0
                continue
            values, (rows, columns) = samples
            values = np.asarray(values)
            rows = np.asarray(rows)
            columns = np.asarray(columns)
            if values.ndim != 1 or rows.ndim != 1 or columns.ndim != 1:
                raise ValueError(f"dataset {split_name} values and indices must be one-dimensional")
            if not (values.size == rows.size == columns.size):
                raise ValueError(f"dataset {split_name} values and indices must have equal length")
            if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(
                columns.dtype, np.integer
            ):
                raise ValueError(f"dataset {split_name} indices must be integers")
            if rows.size and (np.any(rows < 0) or np.any(rows >= self.n_rows)):
                raise ValueError(f"dataset {split_name} row indices are out of bounds")
            if columns.size and (np.any(columns < 0) or np.any(columns >= self.n_cols)):
                raise ValueError(f"dataset {split_name} column indices are out of bounds")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"dataset {split_name} values must be finite")
            split_sizes[split_name] = int(values.size)

        for name, truth_value, expected_shape in (
            ("dense_truth", self.dense_truth, (self.n_rows, self.n_cols)),
            ("latent_truth", self.latent_truth, (self.n_rows + self.n_cols, self.rank)),
            ("linked_truth", self.capabilities.linked_truth, (self.n_rows, self.n_cols)),
        ):
            if truth_value is None:
                continue
            array = np.asarray(truth_value)
            if array.shape != expected_shape:
                raise ValueError(f"dataset {name} must have shape {expected_shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"dataset {name} must be finite")

        splits = self.capabilities.splits
        if splits is not None:
            declared = {
                "train": splits.train_count,
                "validation": splits.validation_count,
                "test": splits.test_count,
            }
            for split_name, count in declared.items():
                if count != split_sizes[split_name]:
                    raise ValueError(
                        f"dataset {split_name} count {count} does not match "
                        f"{split_sizes[split_name]} samples"
                    )

        mappings = self.capabilities.id_mappings
        if mappings is not None:
            for mapping_name, mapping, size in (
                ("row", mappings.row_to_index, self.n_rows),
                ("column", mappings.column_to_index, self.n_cols),
            ):
                indices = list(mapping.values())
                if any(
                    isinstance(index, bool)
                    or not isinstance(index, (int, np.integer))
                    or index < 0
                    or index >= size
                    for index in indices
                ):
                    raise ValueError(f"dataset {mapping_name} ID mapping indices are invalid")
                if len(set(indices)) != len(indices):
                    raise ValueError(f"dataset {mapping_name} ID mapping indices must be unique")


@dataclass
class RunTermination:
    """Why an algorithm stopped and the last completed iteration."""

    reason: str
    iteration: int

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("termination reason must be non-empty")
        if self.iteration < 0:
            raise ValueError("termination iteration must be non-negative")


@dataclass(frozen=True)
class AlgorithmCheckpoint:
    """Versioned, algorithm-owned state captured at one iteration."""

    format_version: int
    algorithm: str
    state_version: int
    elapsed_seconds: float
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version <= 0
        ):
            raise ValueError("checkpoint format_version must be a positive integer")
        if not self.algorithm:
            raise ValueError("checkpoint algorithm must be non-empty")
        if (
            isinstance(self.state_version, bool)
            or not isinstance(self.state_version, int)
            or self.state_version <= 0
        ):
            raise ValueError("checkpoint state_version must be a positive integer")
        if not np.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("checkpoint elapsed_seconds must be finite and non-negative")
        if not isinstance(self.payload, Mapping):
            raise TypeError("checkpoint payload must be a mapping")


@dataclass(frozen=True)
class SelectionPolicy:
    """Choose one coherent observed run state by a diagnostic metric."""

    metric: str
    mode: Literal["min", "max"] = "min"
    retain_checkpoint: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("selection.metric must be a non-empty string")
        if self.mode not in {"min", "max"}:
            raise ValueError("selection.mode must be 'min' or 'max'")
        if not isinstance(self.retain_checkpoint, bool):
            raise ValueError("selection.retain_checkpoint must be a boolean")


@dataclass(frozen=True)
class SelectedRunState:
    """One selected diagnostic record and its optional opaque model state."""

    policy: SelectionPolicy
    iteration: int
    value: float
    diagnostics: Mapping[str, Any]
    checkpoint: AlgorithmCheckpoint | None = None
    exports: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError("selected iteration must be non-negative")
        if not np.isfinite(self.value):
            raise ValueError("selected metric value must be finite")
        if self.policy.retain_checkpoint and self.checkpoint is None:
            raise ValueError("selected state requires a retained checkpoint")


@dataclass
class AlgorithmResult:
    """Stable envelope returned by algorithms."""

    z: np.ndarray
    phi: ScalarFunction | None
    diagnostics: list[dict[str, Any]]
    termination: RunTermination
    checkpoints: dict[int, AlgorithmCheckpoint] = field(default_factory=dict)
    selection: SelectedRunState | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
