from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PlotHook = Callable[["IntegrationCase", Path, Path, Path, dict[str, Any], Path], None]


@dataclass(frozen=True)
class IntegrationCase:
    name: str
    target: str
    executable_relpath: Path
    base_config: Path
    golden_dir: Path
    run_dir: Path
    Ns: int
    tmax: float
    nproc: int
    snapshot_times: tuple[float, ...] = field(default_factory=tuple)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    generate_plots: PlotHook | None = None


DEFAULT_CASE_NAME = "twostream"


from .twostream import CASE as TWOSTREAM_CASE
from .weibel import CASE as WEIBEL_CASE


CASES = {
    case.name: case
    for case in (
        TWOSTREAM_CASE,
        WEIBEL_CASE,
    )
}
