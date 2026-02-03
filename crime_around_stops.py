#!/usr/bin/env python
"""
Visualize TPS MCI crimes near TTC stops (Divisions 14, 51, 52) with folium.

Features
- Loads GTFS stops.txt and TPS MCI CSV/TSV.
- Filters to Divisions 14, 51, 52 (configurable).
- Attaches each crime to the nearest TTC stop within 250 m (haversine; BallTree when available).
- Produces an interactive Leaflet map (via folium) with time-based layers (monthly or yearly).
- Returns the linked crime-stop dataframe and an aggregated dataframe.

Usage (from repo root):
  python crime_around_stops.py \
      --stops "Complete GTFS/stops.txt" \
      --crime "crime.csv" \
      --radius-m 250 \
      --freq M \
      --out crime_around_stops.html
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import folium

EARTH_RADIUS_M = 6_371_000.0
DIVISION_COLORS = {"14": "#1f78b4", "51": "#33a02c", "52": "#e31a1c"}


# Simple stepped scale for stop crime counts
def stop_color_scale(n: float) -> str:
    if n >= 20:
        return "#b10026"  # deep red
    if n >= 10:
        return "#e31a1c"
    if n >= 5:
        return "#fc4e2a"
    if n >= 1:
        return "#fd8d3c"
    return "#c0c0c0"  # light gray for zero


# ----------------------------
# Data loading and validation
# ----------------------------
def load_stops(stops_path: str | pathlib.Path) -> pd.DataFrame:
    """Load GTFS stops.txt and ensure required columns exist."""
    stops_path = pathlib.Path(stops_path)
    stops = pd.read_csv(stops_path)
    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = required - set(stops.columns)
    if missing:
        raise KeyError(f"stops.txt missing columns: {missing}")
    return stops


def _resolve_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def load_stop_times(stop_times_path: str | pathlib.Path) -> pd.DataFrame:
    """
    Load stop_times with optional route info and collapse to per-stop summaries
    to avoid exploding the crime-stop rows.
    """
    stop_times_path = pathlib.Path(stop_times_path)
    if stop_times_path.stat().st_size == 0:
        # Empty file — nothing to enrich with
        return pd.DataFrame()

    try:
        # Force string dtypes on identifiers to avoid mixed-type warnings
        st = pd.read_csv(
            stop_times_path,
            dtype={
                "trip_id": "string",
                "route_id": "string",
                "stop_id": "string",
                "stop_headsign": "string",
            },
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "stop_id" not in st.columns:
        raise KeyError("stop_times file must contain stop_id.")

    gb = st.groupby("stop_id")
    summary = gb.size().rename("stop_time_rows")

    routes = None
    if "route_id" in st.columns:
        routes = gb["route_id"].agg(lambda s: ",".join(sorted(pd.unique(s.dropna().astype(str))))).rename("routes")

    trip_count = None
    if "trip_id" in st.columns:
        trip_count = gb["trip_id"].nunique().rename("trip_count")

    df = summary.to_frame()
    if routes is not None:
        df = df.join(routes)
    if trip_count is not None:
        df = df.join(trip_count)

    return df.reset_index()


def load_crime(crime_path: str | pathlib.Path) -> pd.DataFrame:
    """
    Load TPS crime CSV/TSV. Expected fields (flexible names):
    - occurrence_date / occurrence_datetime / occ_date / report_date
    - lat_wgs84 / latitude / lat
    - long_wgs84 / longitude / lon / lng
    - division
    - MCI or a general crime type column (kept if present)
    """
    crime_path = pathlib.Path(crime_path)
    
    # Check if file is a Git LFS pointer (Streamlit Cloud doesn't support LFS)
    # Skip check for compressed files - they can't be LFS pointers
    if crime_path.exists() and not str(crime_path).endswith(('.gz', '.zip', '.bz2', '.xz')):
        try:
            with open(crime_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line == "version https://git-lfs.github.com/spec/v1":
                    # Return empty DataFrame instead of raising error - let UI handle the message
                    return pd.DataFrame()
        except UnicodeDecodeError:
            # If we can't decode as UTF-8, it's not a text LFS pointer, continue normally
            pass
    
    # pandas automatically handles .gz files
    try:
        crime = pd.read_csv(crime_path)
    except Exception as e:
        # If reading fails (e.g., corrupted file, wrong format), return empty DataFrame
        return pd.DataFrame()
    
    # Process the loaded data
    lat_col = _resolve_col(crime, ["lat_wgs84", "latitude", "lat"])
    lon_col = _resolve_col(crime, ["long_wgs84", "longitude", "lon", "lng"])
    div_col = _resolve_col(crime, ["division"])
    dt_col = _resolve_col(
        crime,
        [
            "occurrence_datetime",
            "occurrence_date",
            "occ_date",
            "report_date",  # TPS open data export
        ],
    )

    crime = crime.rename(
        columns={
            lat_col: "lat",
            lon_col: "lon",
            div_col: "division",
            dt_col: "crime_dt",
        }
    )
    crime["crime_dt"] = pd.to_datetime(crime["crime_dt"], errors="coerce")
    crime = crime[crime["crime_dt"].notna()].copy()
    # Normalize division codes: accept formats like "D14" or "14"
    crime["division"] = (
        crime["division"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(2)
    )
    return crime


def load_event_days(calendar_dates_path: str | pathlib.Path) -> set[datetime.date]:
    """
    Load GTFS calendar_dates.txt and return a set of event/special-service dates (as datetime.date).
    Expects a 'date' column in YYYYMMDD format.
    """
    cal = pd.read_csv(calendar_dates_path)
    if "date" not in cal.columns:
        raise KeyError("calendar_dates.txt must contain 'date' in YYYYMMDD format.")
    return set(pd.to_datetime(cal["date"], format="%Y%m%d").dt.date)


# ----------------------------
# Distance + nearest stop
# ----------------------------
def haversine_rad(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (meters) with radian inputs; broadcast friendly."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def attach_nearest_stop(
    crime_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    max_dist_m: float = 250.0,
) -> pd.DataFrame:
    """
    Attach nearest stop within max_dist_m. Falls back to pure numpy if BallTree unavailable.
    """
    stops_rad = np.deg2rad(stops_df[["stop_lat", "stop_lon"]].to_numpy())
    crimes_rad = np.deg2rad(crime_df[["lat", "lon"]].to_numpy())

    try:
        from sklearn.neighbors import BallTree  # type: ignore

        tree = BallTree(stops_rad, metric="haversine")
        dist_rad, idx = tree.query(crimes_rad, k=1)
        dist_m = dist_rad[:, 0] * EARTH_RADIUS_M
        nearest_idx = idx[:, 0]
    except Exception:
        # Chunked numpy fallback to control memory
        batch = 5000
        dist_m = np.empty(len(crime_df))
        nearest_idx = np.empty(len(crime_df), dtype=int)
        for start in range(0, len(crime_df), batch):
            end = min(start + batch, len(crime_df))
            chunk = crimes_rad[start:end][:, None, :]  # (b,1,2)
            dist = haversine_rad(chunk[..., 0], chunk[..., 1], stops_rad[:, 0], stops_rad[:, 1])
            nearest_idx[start:end] = dist.argmin(axis=1)
            dist_m[start:end] = dist.min(axis=1)

    out = crime_df.copy()
    out["nearest_stop_id"] = stops_df.iloc[nearest_idx]["stop_id"].to_numpy()
    out["nearest_stop_name"] = stops_df.iloc[nearest_idx]["stop_name"].to_numpy()
    out["dist_m"] = dist_m
    out = out[out["dist_m"] <= max_dist_m].copy()
    return out


# ----------------------------
# Aggregation + mapping
# ----------------------------
def add_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df["period"] = df["crime_dt"].dt.to_period(freq).dt.to_timestamp()
    return df


def aggregate(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    df = add_period(df, freq)
    agg = (
        df.groupby(["period", "division", "nearest_stop_id", "nearest_stop_name"])
        .size()
        .reset_index(name="crime_count")
    )
    return agg


def make_map(
    linked: pd.DataFrame,
    stops: pd.DataFrame,
    freq: str = "M",
    radius_m: float = 250.0,
    risk_lookup: dict[str, float] | None = None,
    risk_level_lookup: dict[str, str] | None = None,
    daily_avg_lookup: dict[str, float] | None = None,
    event_daily_avg_lookup: dict[str, float] | None = None,
    normal_daily_avg_lookup: dict[str, float] | None = None,
    majority_type_lookup: dict[str, str] | None = None,
    trend_color_lookup: dict[str, str] | None = None,
    trend_text_lookup: dict[str, str] | None = None,
    spike_color_lookup: dict[str, str] | None = None,
    spike_blink_lookup: dict[str, bool] | None = None,
    spike_text_lookup: dict[str, str] | None = None,
) -> folium.Map:
    # Base map centered on downtown Toronto
    m = folium.Map(location=[43.6532, -79.3832], zoom_start=12, tiles="CartoDB positron")

    # Crimes + cumulative stop counts per period layer
    linked = add_period(linked, freq)
    stop_lookup = stops.set_index("stop_id")

    # Stops: color by risk level when available
    level_colors = {
        "Low": "#9e9e9e",        # gray
        "Moderate": "#1f78b4",   # blue
        "Elevated": "#ff9800",   # orange
        "High": "#7b1fa2",       # purple
    }

    def stop_color(level: str | None) -> str:
        return level_colors.get(level, "#000000")

    # Sort periods chronologically for cumulative roll-up
    periods_sorted = sorted(linked["period"].unique())
    cumulative_counts: dict[str, int] = {sid: 0 for sid in stop_lookup.index}

    for i, period in enumerate(periods_sorted):
        grp = linked[linked["period"] == period]
        layer = folium.FeatureGroup(
            name=f"through {period.strftime('%Y-%m') if freq=='M' else period.year}",
            show=True,  # show all periods so full date range is visible
        )

        # Update cumulative counts
        increment = grp.groupby("nearest_stop_id").size()
        for stop_id, c in increment.items():
            cumulative_counts[stop_id] = cumulative_counts.get(stop_id, 0) + int(c)

        # Period-specific counts for display
        period_counts = increment.to_dict()

        # Draw stop squares with cumulative counts
        for stop_id, total in cumulative_counts.items():
            if stop_id not in stop_lookup.index:
                continue
            row = stop_lookup.loc[stop_id]
            score = None
            level = None
            if risk_lookup:
                score = risk_lookup.get(str(stop_id)) or risk_lookup.get(stop_id)
            if risk_level_lookup:
                level = risk_level_lookup.get(str(stop_id)) or risk_level_lookup.get(stop_id)
            daily_avg = None
            event_avg = None
            normal_avg = None
            maj_type = None
            if daily_avg_lookup:
                daily_avg = daily_avg_lookup.get(str(stop_id)) or daily_avg_lookup.get(stop_id)
            if event_daily_avg_lookup:
                event_avg = event_daily_avg_lookup.get(str(stop_id)) or event_daily_avg_lookup.get(stop_id)
            if normal_daily_avg_lookup:
                normal_avg = normal_daily_avg_lookup.get(str(stop_id)) or normal_daily_avg_lookup.get(stop_id)
            if majority_type_lookup:
                maj_type = majority_type_lookup.get(str(stop_id)) or majority_type_lookup.get(stop_id)
            trend_color = None
            trend_text = None
            if trend_color_lookup:
                trend_color = trend_color_lookup.get(str(stop_id)) or trend_color_lookup.get(stop_id)
            if trend_text_lookup:
                trend_text = trend_text_lookup.get(str(stop_id)) or trend_text_lookup.get(stop_id)
            spike_color = None
            spike_blink = False
            spike_text = None
            if spike_color_lookup:
                spike_color = spike_color_lookup.get(str(stop_id)) or spike_color_lookup.get(stop_id)
            if spike_blink_lookup:
                spike_blink = spike_blink_lookup.get(str(stop_id)) or spike_blink_lookup.get(stop_id)
            if spike_text_lookup:
                spike_text = spike_text_lookup.get(str(stop_id)) or spike_text_lookup.get(stop_id)
            popup_html_parts = [
                f"<b>{row.stop_name}</b><br>",
                f"<span style='color:#666'>Stop ID:</span> {stop_id}<br>",
                "<span style='color:#666'>Crimes through:</span> "
                f"{period.strftime('%B %Y') if freq=='M' else period.year}<br>",
                f"<span style='color:#666'>Crimes in this layer:</span> {period_counts.get(stop_id, 0)}<br>",
                f"<span style='color:#666'>Total linked crimes (filtered):</span> {total}",
            ]
            if daily_avg is not None:
                popup_html_parts.append(f"<br><span style='color:#666'>Avg crimes/day (filtered):</span> {daily_avg:.2f}")
            if event_avg is not None:
                popup_html_parts.append(f"<br><span style='color:#666'>Event-day avg:</span> {event_avg:.2f}")
            if normal_avg is not None:
                popup_html_parts.append(f"<br><span style='color:#666'>Normal-day avg:</span> {normal_avg:.2f}")
            if score is not None:
                popup_html_parts.append(f"<br><span style='color:#666'>Risk score:</span> {score:.3f}")
            if level is not None:
                popup_html_parts.append(f"<br><span style='color:#666'>Risk level:</span> {level}")
            if maj_type is not None:
                popup_html_parts.append(f"<br><span style='color:#666'>Top crime type:</span> {maj_type}")
            if trend_text is not None:
                popup_html_parts.append(f"<br><span style='color:#666'>Rolling trend:</span><br>{trend_text}")
            if spike_text is not None:
                popup_html_parts.append(
                    f"<br><span style='color:#666'>2hr spike:</span> {spike_text}"
                    + (" (blink)" if spike_blink else "")
                )

            folium.RegularPolygonMarker(
                location=[row.stop_lat, row.stop_lon],
                number_of_sides=4,
                radius=min(8, 2 + (total ** 0.5) / 2),
                color=stop_color(level),
                fill=True,
                fill_opacity=0.9,
                popup=folium.Popup(html="".join(popup_html_parts), max_width=260),
            ).add_to(layer)

            # Rolling trend overlay marker (color shows expectedRisk classification).
            if trend_color:
                folium.CircleMarker(
                    location=[row.stop_lat, row.stop_lon],
                    radius=4.5,
                    color=trend_color,
                    fill=True,
                    fill_color=trend_color,
                    fill_opacity=0.85,
                    weight=2,
                ).add_to(layer)

            # Spike detection marker: dark gray = spike, light gray = no spike. Prominent ring + light fill.
            if spike_color:
                folium.CircleMarker(
                    location=[row.stop_lat, row.stop_lon],
                    radius=20,
                    color=spike_color,
                    fill=True,
                    fill_color=spike_color,
                    fill_opacity=0.25,
                    weight=6,
                    popup=folium.Popup(
                        html=f"<b>2hr TPS rule</b><br>{spike_text or 'N/A'}"
                        + ("<br><i>Blink in live UI</i>" if spike_blink else ""),
                        max_width=220,
                    ),
                ).add_to(layer)

        # Individual crimes for this period
        for _, row in grp.iterrows():
            color = DIVISION_COLORS.get(row.division, "#444")
            parts = [
                "<div style='font-size:12px; line-height:1.35em; max-width:220px;'>",
                f"<div style='font-size:13px; font-weight:600; margin-bottom:4px;'>{row.get('MCI', row.get('crime_type', 'Crime'))}</div>",
                f"<div><span style='color:#666'>Division:</span> {row.division}</div>",
                f"<div><span style='color:#666'>Date:</span> {row.crime_dt.date()}</div>",
                f"<div><span style='color:#666'>Nearest stop:</span><br>{row.nearest_stop_name} <span style='color:#888'>({row.nearest_stop_id})</span></div>",
                f"<div><span style='color:#666'>Distance:</span> {row.dist_m:.0f} m</div>",
                "</div>",
            ]
            popup = folium.Popup(html="".join(parts), max_width=260)
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup,
            ).add_to(layer)

        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend for risk levels (if provided)
    if risk_level_lookup:
        legend_html = """
        <div style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 9999;
            background: white;
            padding: 10px 12px;
            border: 1px solid #ccc;
            box-shadow: 0 0 6px rgba(0,0,0,0.25);
            font-size: 12px;
            line-height: 1.4;
        ">
          <b>Risk level</b><br>
          <i style="background:#9e9e9e; width:12px; height:12px; display:inline-block; margin-right:6px;"></i>Low<br>
          <i style="background:#1f78b4; width:12px; height:12px; display:inline-block; margin-right:6px;"></i>Moderate<br>
          <i style="background:#ff9800; width:12px; height:12px; display:inline-block; margin-right:6px;"></i>Elevated<br>
          <i style="background:#7b1fa2; width:12px; height:12px; display:inline-block; margin-right:6px;"></i>High<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    # Legend for rolling trend colors (if provided)
    if trend_color_lookup:
        trend_legend_html = """
        <div style="
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            background: white;
            padding: 10px 12px;
            border: 1px solid #ccc;
            box-shadow: 0 0 6px rgba(0,0,0,0.25);
            font-size: 12px;
            line-height: 1.4;
        ">
          <b>Rolling trend</b><br>
          <i style="background:#F9A825; width:12px; height:12px; display:inline-block; margin-right:6px;"></i>HIGH (delta &gt; 1.15)<br>
          <i style="background:#FDD835; width:12px; height:12px; display:inline-block; margin-right:6px;"></i>STABLE (0.9–1.15)<br>
          <i style="background:#FFF9C4; width:12px; height:12px; display:inline-block; margin-right:6px; border:1px solid #ddd;"></i>DECREASING (delta &lt; 0.9)<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(trend_legend_html))

    # Legend for 2hr spike detection (if provided); place above risk legend; gray shades
    if spike_color_lookup:
        spike_legend_html = """
        <div style="
            position: fixed;
            bottom: 120px;
            left: 20px;
            z-index: 9998;
            background: white;
            padding: 10px 12px;
            border: 1px solid #ccc;
            box-shadow: 0 0 6px rgba(0,0,0,0.25);
            font-size: 12px;
            line-height: 1.4;
        ">
          <b>2hr spike (TPS rule)</b><br>
          <i style="width:14px; height:14px; display:inline-block; margin-right:6px; background:#37474f; border:2px solid #263238; border-radius:50%; box-sizing:border-box;"></i>Spike (blink)<br>
          <i style="width:14px; height:14px; display:inline-block; margin-right:6px; background:#90a4ae; border:2px solid #78909c; border-radius:50%; box-sizing:border-box;"></i>No spike<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(spike_legend_html))

    return m


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map TPS crimes near TTC stops.")
    p.add_argument("--stops", default="Complete GTFS/stops.txt", help="Path to GTFS stops.txt")
    p.add_argument(
        "--stop-times",
        default="data/processed/stop_times_with_stops.csv.gz",
        help="Path to stop_times file (used to enrich stops with routes/trip counts)",
    )
    # Default to the local TPS MCI export if present
    p.add_argument(
        "--crime",
        default="Major_Crime_Indicators.csv",
        help="Path to TPS crime CSV/TSV (default: Major_Crime_Indicators.csv in repo root)",
    )
    p.add_argument("--out", default="crime_around_stops.html", help="Output HTML map path")
    p.add_argument("--radius-m", type=float, default=250.0, help="Max distance to link crimes to a stop (meters)")
    p.add_argument("--freq", default="M", choices=["M", "Y"], help="Time aggregation: M=monthly, Y=yearly")
    p.add_argument(
        "--divisions",
        default="14,51,52",
        help="Comma-separated division codes to include (default: 14,51,52)",
    )
    p.add_argument(
        "--years-back",
        type=int,
        default=5,
        help="Keep crimes within the last N years from today (default: 5)",
    )
    p.add_argument("--linked-out", help="Optional path to save linked crime-stop CSV")
    p.add_argument("--agg-out", help="Optional path to save aggregated CSV")
    return p.parse_args()


def main():
    args = parse_args()

    stops = load_stops(args.stops)
    crime = load_crime(args.crime)
    stop_times_summary = (
        load_stop_times(args.stop_times) if pathlib.Path(args.stop_times).exists() else None
    )

    # Date filter: keep only crimes in the last N years
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=args.years_back)
    crime = crime[crime["crime_dt"] >= cutoff]

    divisions = [d.strip() for d in args.divisions.split(",") if d.strip()]
    crime = crime[crime["division"].isin(divisions)].copy()

    linked = attach_nearest_stop(crime, stops, max_dist_m=args.radius_m)

    # Only keep stops that have at least one crime in the selected divisions
    stops_for_map = stops[stops["stop_id"].isin(linked["nearest_stop_id"])].copy()

    if stop_times_summary is not None and not stop_times_summary.empty:
        linked = linked.merge(
            stop_times_summary,
            left_on="nearest_stop_id",
            right_on="stop_id",
            how="left",
            suffixes=("", "_stoptime"),
        )

    agg = aggregate(linked, freq=args.freq)
    fmap = make_map(linked, stops_for_map, freq=args.freq, radius_m=args.radius_m)
    fmap.save(args.out)

    if args.linked_out:
        linked.to_csv(args.linked_out, index=False)
    if args.agg_out:
        agg.to_csv(args.agg_out, index=False)

    # Event vs Normal day aggregations using calendar_dates.txt if present
    cal_path = pathlib.Path("Complete GTFS/calendar_dates.txt")
    if cal_path.exists():
        event_days = load_event_days(cal_path)
        # Label crimes
        linked["date"] = linked["crime_dt"].dt.date
        linked["label"] = np.where(linked["date"].isin(event_days), "Event day", "Normal day")
        linked["hour"] = linked["crime_dt"].dt.hour

        # Per-day counts by stop/label
        daily_by_stop = (
            linked.groupby(["nearest_stop_id", "nearest_stop_name", "label", "date"])
            .size()
            .rename("count")
            .reset_index()
        )

        # Average crimes per day by stop/label
        stop_label_avg = (
            daily_by_stop.groupby(["nearest_stop_id", "nearest_stop_name", "label"])["count"]
            .mean()
            .rename("avg_crimes_per_day")
            .reset_index()
        )

        # Violent proportion (heuristic categories)
        violent_cats = {"Assault", "Robbery", "Homicide", "Sexual Violation"}
        if "MCI_CATEGORY" in linked.columns:
            linked["is_violent"] = linked["MCI_CATEGORY"].isin(violent_cats)
            violent_prop = (
                linked.groupby(["nearest_stop_id", "nearest_stop_name", "label"])["is_violent"]
                .mean()
                .rename("violent_prop")
                .reset_index()
            )
            summary = stop_label_avg.merge(violent_prop, on=["nearest_stop_id", "nearest_stop_name", "label"], how="left")
        else:
            summary = stop_label_avg

        summary.to_csv("summary_event_vs_normal.csv", index=False)
        daily_by_stop.to_csv("daily_counts_event_vs_normal.csv", index=False)

        print("Wrote summary_event_vs_normal.csv and daily_counts_event_vs_normal.csv")
    else:
        print("calendar_dates.txt not found; skipping event-vs-normal aggregation.")

    print(f"Map written to {args.out}")
    print(f"Linked rows: {len(linked)}; Aggregated groups: {len(agg)}")
    return fmap, linked, agg


if __name__ == "__main__":
    main()
