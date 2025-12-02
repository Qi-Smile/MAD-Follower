"""
Placeholder visualization module.
Existing plotting helpers live in scripts/run_analysis_demo.py; this module
provides a future home for reusable plotting logic to avoid duplication across
scripts. Importers should migrate plots here over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# TODO: Move plot_drift_metrics, plot_answer_changes, plot_conformity_rate,
# plot_confidence_trend from scripts/run_analysis_demo.py into this module.


def not_implemented(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("Visualization functions are not yet centralized; use scripts/run_analysis_demo.py plots.")


plot_drift_metrics = not_implemented
plot_answer_changes = not_implemented
plot_conformity_rate = not_implemented
plot_confidence_trend = not_implemented
