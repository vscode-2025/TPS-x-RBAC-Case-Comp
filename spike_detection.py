"""Spike detection helpers for the TPS two-hour surge rule.

TPS rule:
  If crime count in the recent 2-hour window exceeds the historical 2-hour baseline
  by >30%, trigger +2 patrol units.

Definitions:
  baseline_2hr(station) = average crimes per 2-hour window across the historical horizon
  recent_2hr(station)   = crimes in the trailing 2-hour window ending at a chosen timestamp

Implementation notes:
- baseline_2hr must include 2-hour windows with zero crimes, otherwise the baseline is biased high.
- We therefore compute baseline as:
    baseline = total_crimes_in_period / number_of_2h_windows_in_period
  (per station, using a shared window count across the analysis period).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd

TWO_HOURS = pd.Timedelta(hours=2)

__all__ = [
    "compute_baseline_2hr",
    "compute_recent_2hr",
    "detect_spike",
    "build_station_snapshot",
    "build_example_outputs",
]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing columns: {missing}")


def _prepare_frame(df: pd.DataFrame, station_col: str, ts_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[station_col, ts_col])
    _ensure_columns(df, [station_col, ts_col])
    out = df[[station_col, ts_col]].dropna().copy()
    if out.empty:
        return out
    out[station_col] = out[station_col].astype(str)
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    return out.dropna(subset=[ts_col])


def compute_baseline_2hr(
    df: pd.DataFrame,
    *,
    station_col: str = "station_id",
    ts_col: str = "timestamp",
    as_of: Any | None = None,
) -> pd.Series:
    """Compute baseline_2hr per station.

    baseline_2hr(station) = total_crimes(station) / num_2hour_windows

    If as_of is provided, only history with timestamps <= as_of is used,
    preventing "future leakage" when the user picks an earlier reference time.
    """
    prepared = _prepare_frame(df, station_col, ts_col)
    if prepared.empty:
        return pd.Series(dtype=float, name="baseline_2hr")

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        prepared = prepared[prepared[ts_col] <= as_of_ts]
        if prepared.empty:
            return pd.Series(dtype=float, name="baseline_2hr")

    start = prepared[ts_col].min()
    end = prepared[ts_col].max()
    if pd.isna(start) or pd.isna(end) or end <= start:
        return pd.Series(dtype=float, name="baseline_2hr")

    # Align to a stable 2-hour grid.
    start_aligned = pd.Timestamp(start).floor(freq="2H")
    end_aligned = pd.Timestamp(end).ceil(freq="2H")
    num_windows = int((end_aligned - start_aligned) / TWO_HOURS)
    num_windows = max(num_windows, 1)

    totals = prepared.groupby(station_col).size().astype("int64")
    baseline = (totals / float(num_windows)).astype("float64").rename("baseline_2hr")
    return baseline


def compute_recent_2hr(
    df: pd.DataFrame,
    *,
    as_of: Any,
    station_col: str = "station_id",
    ts_col: str = "timestamp",
) -> pd.Series:
    """Compute recent_2hr per station in (as_of - 2h, as_of]."""
    if as_of is None:
        raise ValueError("as_of timestamp is required")
    prepared = _prepare_frame(df, station_col, ts_col)
    if prepared.empty:
        return pd.Series(dtype=float, name="recent_2hr")

    as_of_ts = pd.Timestamp(as_of)
    window_start = as_of_ts - TWO_HOURS
    windowed = prepared[(prepared[ts_col] > window_start) & (prepared[ts_col] <= as_of_ts)]
    if windowed.empty:
        return pd.Series(dtype=float, name="recent_2hr")
    return windowed.groupby(station_col).size().astype("float64").rename("recent_2hr")


def _ui_flag(spike: bool) -> dict[str, Any]:
    return {"color": "red", "blink": True} if spike else {"color": "blue", "blink": False}


def detect_spike(
    station_id: str,
    *,
    recent_2hr: float,
    baseline_2hr: float,
    threshold_pct: float = 0.30,
    percentage_precision: int = 0,
) -> dict[str, Any]:
    """Return the spike payload for one station."""
    baseline_val = max(float(baseline_2hr), 0.0)
    recent_val = max(float(recent_2hr), 0.0)

    if baseline_val == 0.0:
        # Undefined percentage above baseline; use 100% when any crimes are observed.
        percentage_above = 100.0 if recent_val > 0 else 0.0
        trigger = recent_val > 0
    else:
        percentage_above = max(((recent_val - baseline_val) / baseline_val) * 100.0, 0.0)
        trigger = recent_val > baseline_val * (1.0 + float(threshold_pct))  # strict "exceeds by >30%"

    percentage_above = round(float(percentage_above), int(percentage_precision))
    flag = _ui_flag(trigger)

    return {
        "station_id": str(station_id),
        "baseline_2hr": round(baseline_val, 2),
        "recent_2hr": round(recent_val, 2),
        "percentageAbove": percentage_above,
        "spike": bool(trigger),
        "trigger": bool(trigger),
        "recommended_units": 2 if trigger else 0,
        "ui_flag": flag,
    }


def build_station_snapshot(
    df: pd.DataFrame,
    *,
    as_of: Any,
    station_col: str = "station_id",
    ts_col: str = "timestamp",
    station_filter: Sequence[str] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build spike payloads for all stations (optionally filtered)."""
    baselines = compute_baseline_2hr(df, station_col=station_col, ts_col=ts_col, as_of=as_of)
    recents = compute_recent_2hr(df, as_of=as_of, station_col=station_col, ts_col=ts_col)

    baseline_ids = set(baselines.index.astype(str)) if not baselines.empty else set()
    recent_ids = set(recents.index.astype(str)) if not recents.empty else set()
    station_ids = baseline_ids.union(recent_ids)

    if station_filter:
        wanted = {str(s) for s in station_filter}
        station_ids = station_ids.intersection(wanted)
        if not station_ids:
            station_ids = wanted

    payloads: list[dict[str, Any]] = []
    for sid in sorted(station_ids):
        payloads.append(
            detect_spike(
                sid,
                recent_2hr=float(recents.get(sid, 0.0)) if not recents.empty else 0.0,
                baseline_2hr=float(baselines.get(sid, 0.0)) if not baselines.empty else 0.0,
            )
        )

    # Show biggest spikes first (percentageAbove desc, then recent_2hr desc).
    return sorted(payloads, key=lambda x: (x.get("percentageAbove", 0), x.get("recent_2hr", 0)), reverse=True)


def build_example_outputs() -> list[dict[str, Any]]:
    """Example output for 3 stations: no spike / borderline (31%) / strong (60%)."""
    examples = [
        ("ST-101", 5.0, 5.0),   # no spike
        ("ST-204", 4.5, 5.9),   # ~31% above baseline
        ("ST-317", 3.0, 4.8),   # 60% above baseline
    ]
    return [
        detect_spike(station, recent_2hr=recent, baseline_2hr=baseline)
        for station, baseline, recent in examples
    ]

