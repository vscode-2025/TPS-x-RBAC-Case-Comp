"""Rolling trend risk analysis (30-day vs 60-day).

This module implements a lightweight "short-term vs mid-term" comparison for
crime volume per station.

Definitions (per station):
- rolling30: average daily crimes over last 30 days
- rolling60: average daily crimes over last 60 days
- delta: rolling30 / rolling60

Classification (colors requested):
- delta > 1.15  -> HIGH (dark yellow)
- 0.9..1.15     -> STABLE (medium yellow)
- delta < 0.9   -> DECREASING (light yellow)

The functions are written to be:
- modular (usable in Streamlit, API, or batch jobs)
- correct with missing days (fills zeros so averages are unbiased)
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd

WINDOW_30_DAYS = 30
WINDOW_60_DAYS = 60

__all__ = [
    "compute_daily_counts",
    "compute_rolling_averages",
    "classify_trend",
    "build_trend_payload",
    "build_trend_snapshot",
    "build_trend_example_outputs",
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


def compute_daily_counts(
    df: pd.DataFrame,
    *,
    station_col: str = "station_id",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Return daily crime counts per station.

    Output columns:
    - station_id (string)
    - date (Timestamp normalized to midnight)
    - count (int)
    """
    prepared = _prepare_frame(df, station_col, ts_col)
    if prepared.empty:
        return pd.DataFrame(columns=["station_id", "date", "count"])

    daily = prepared.copy()
    daily["date"] = daily[ts_col].dt.normalize()
    counts = (
        daily.groupby([station_col, "date"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={station_col: "station_id"})
    )
    counts["station_id"] = counts["station_id"].astype(str)
    counts["count"] = counts["count"].astype(int)
    return counts


def compute_rolling_averages(
    daily_counts: pd.DataFrame,
    *,
    as_of: Any,
    window30: int = WINDOW_30_DAYS,
    window60: int = WINDOW_60_DAYS,
) -> pd.DataFrame:
    """Compute rolling30 and rolling60 per station as of a reference date.

    Important: this fills missing days with 0 crimes so the averages represent
    true "per-day" averages over the full trailing window.
    """
    if daily_counts.empty:
        return pd.DataFrame(columns=["station_id", "rolling30", "rolling60"])
    _ensure_columns(daily_counts, ["station_id", "date", "count"])

    as_of_date = pd.Timestamp(as_of).normalize()
    trailing_days = max(int(window30), int(window60))
    date_index = pd.date_range(end=as_of_date, periods=trailing_days, freq="D")

    results: list[dict[str, float | str]] = []
    for station_id, group in daily_counts.groupby("station_id"):
        series = group.set_index("date")["count"].astype(float)
        series = series.reindex(date_index, fill_value=0.0)
        rolling30 = float(series.tail(window30).mean())
        rolling60 = float(series.tail(window60).mean())
        results.append({"station_id": str(station_id), "rolling30": rolling30, "rolling60": rolling60})

    return pd.DataFrame(results)


def classify_trend(delta: float) -> tuple[str, str]:
    """Return (expectedRisk, map_color) from delta."""
    if delta > 1.15:
        return "HIGH", "#F9A825"  # dark yellow (amber)
    if 0.9 <= delta <= 1.15:
        return "STABLE", "#FDD835"  # medium yellow
    return "DECREASING", "#FFF9C4"  # light yellow


def build_trend_payload(
    station_id: str,
    rolling30: float,
    rolling60: float,
) -> dict[str, Any]:
    """Build the JSON output for one station (matches the spec)."""
    r30 = max(float(rolling30), 0.0)
    r60 = max(float(rolling60), 0.0)

    if r60 == 0.0:
        delta = float("inf") if r30 > 0 else 1.0
    else:
        delta = r30 / r60

    expected_risk, color = classify_trend(delta if delta != float("inf") else 9999.0)

    if delta == float("inf"):
        trend_text = "Recent Trend: baseline near 0 (30-day vs 60-day)"
    else:
        pct = (delta - 1.0) * 100.0
        if pct >= 0:
            trend_text = f"Recent Trend: +{pct:.0f}% increase (30-day vs 60-day)"
        else:
            trend_text = f"Recent Trend: {pct:.0f}% decrease (30-day vs 60-day)"

    if expected_risk == "HIGH":
        recommendation = "Allocate +1 unit tonight (based on upward trend)"
    elif expected_risk == "DECREASING":
        recommendation = "Consider reallocating resources elsewhere (downward trend)"
    else:
        recommendation = "Maintain current patrol allocation (trend stable)"

    return {
        "station_id": str(station_id),
        "rolling30": round(r30, 2),
        "rolling60": round(r60, 2),
        "delta": round(delta, 2) if delta != float("inf") else delta,
        "trendText": trend_text,
        "expectedRisk": expected_risk,
        "recommendation": recommendation,
        # Map UI behavior per spec
        "map_flag": {"color": color},
    }


def build_trend_snapshot(
    df: pd.DataFrame,
    *,
    as_of: Any | None = None,
    station_col: str = "station_id",
    ts_col: str = "timestamp",
    station_filter: Sequence[str] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute the rolling-trend payloads for all stations (optionally filtered)."""
    prepared = _prepare_frame(df, station_col, ts_col)
    if prepared.empty:
        return []

    # Reference date defaults to the newest day in the dataset.
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else prepared[ts_col].max()
    as_of_ts = pd.Timestamp(as_of_ts).normalize()

    daily = compute_daily_counts(prepared, station_col=station_col, ts_col=ts_col)
    avgs = compute_rolling_averages(daily, as_of=as_of_ts)
    if avgs.empty:
        return []

    if station_filter:
        wanted = {str(s) for s in station_filter}
        avgs = avgs[avgs["station_id"].astype(str).isin(wanted)]

    payloads = [
        build_trend_payload(row.station_id, row.rolling30, row.rolling60)
        for row in avgs.itertuples(index=False)
    ]
    # Highest delta first (most increasing)
    return sorted(payloads, key=lambda x: (x["delta"] if x["delta"] != float("inf") else 10_000.0), reverse=True)


def build_trend_example_outputs() -> list[dict[str, Any]]:
    """Example outputs for 3 stations (HIGH / STABLE / DECREASING)."""
    return [
        build_trend_payload("ST-101", rolling30=4.8, rolling60=4.0),   # delta=1.20 HIGH
        build_trend_payload("ST-204", rolling30=4.1, rolling60=4.0),   # delta=1.02 STABLE
        build_trend_payload("ST-317", rolling30=3.2, rolling60=4.0),   # delta=0.80 DECREASING
    ]

