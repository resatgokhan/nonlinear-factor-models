"""Grouped diagnostic families and registry."""

from .data_summary import DataSummaryDiagnostics
from .gradients import GradientDiagnostics
from .objective import ObjectiveDiagnostics
from .phi import PhiDiagnostics
from .phi_sharp import PhiSharpDiagnostics
from .reconstruction import ReconstructionDiagnostics
from .registry import (
    DiagnosticComputationError,
    DiagnosticContext,
    DiagnosticFailure,
    DiagnosticFamily,
    DiagnosticReport,
    DiagnosticRequest,
    compute_requested_diagnostics,
    get_diagnostic_family,
    list_diagnostic_families,
    register_diagnostic_family,
)
from .snapshot import ObservedSplit, RunSnapshot, readonly_array
from .timing import TimingDiagnostics
from .truth_matrix import TruthMatrixDiagnostics
from .truth_z import TruthZDiagnostics

# Built-ins are wired here in one visible, deterministic order. External
# families use the public register_diagnostic_family() API.
register_diagnostic_family(DataSummaryDiagnostics())
register_diagnostic_family(GradientDiagnostics())
register_diagnostic_family(ObjectiveDiagnostics())
register_diagnostic_family(PhiDiagnostics())
register_diagnostic_family(PhiSharpDiagnostics())
register_diagnostic_family(ReconstructionDiagnostics())
register_diagnostic_family(TimingDiagnostics())
register_diagnostic_family(TruthMatrixDiagnostics())
register_diagnostic_family(TruthZDiagnostics())

__all__ = [
    "DiagnosticContext",
    "DiagnosticComputationError",
    "DiagnosticFailure",
    "DiagnosticFamily",
    "DiagnosticReport",
    "DiagnosticRequest",
    "compute_requested_diagnostics",
    "get_diagnostic_family",
    "list_diagnostic_families",
    "register_diagnostic_family",
    "ObservedSplit",
    "RunSnapshot",
    "readonly_array",
]
