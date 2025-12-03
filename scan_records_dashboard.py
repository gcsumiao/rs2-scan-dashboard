#!/usr/bin/env python3
"""Build an interactive HTML dashboard from scan_records.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and normalize the scans CSV."""
    df = pd.read_csv(csv_path)
    df["CreatedDateTimeUTC"] = pd.to_datetime(df["CreatedDateTimeUTC"], errors="coerce")

    # Normalize identifiers and state codes.
    df["AccountId"] = df["AccountId"].astype(str).str.strip()
    df["Email"] = df["Email"].astype(str).str.strip()
    df["VIN"] = df["VIN"].astype(str).str.strip()
    df["State"] = df["State"].astype(str).str.strip().str.upper()
    df.loc[~df["State"].str.fullmatch(r"[A-Z]{2}"), "State"] = pd.NA

    # Prefer AccountId, fall back to Email if AccountId is missing.
    df["AccountKey"] = df["AccountId"].where(df["AccountId"].ne(""), df["Email"])
    return df


class Metric(TypedDict):
    label: str
    value: float
    description: str
    kind: str  # "percent" or "count"


def summarize(df: pd.DataFrame) -> Tuple[List[Metric], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute key summary tables and series."""
    total_scans = len(df)

    user_counts = df.groupby("AccountKey").size()
    repeat_user_mask = user_counts > 1
    repeat_user_ids = user_counts[repeat_user_mask].index

    repeat_user_scans = int(user_counts[repeat_user_mask].sum())
    repeat_user_scan_share = repeat_user_scans / total_scans if total_scans else 0

    unique_users = int(user_counts.size)
    repeat_users = int(repeat_user_mask.sum())
    repeat_user_share = repeat_users / unique_users if unique_users else 0

    vin_counts = df["VIN"].dropna().value_counts()
    repeat_vins = vin_counts > 1

    summary: List[Metric] = [
        {
            "label": "Total scans",
            "value": total_scans,
            "description": "All scan events in the dataset.",
            "kind": "count",
        },
        {
            "label": "Repeat-user scans",
            "value": repeat_user_scans,
            "description": "Scan events coming from repeat users.",
            "kind": "count",
        },
        {
            "label": "Repeat-user scan share",
            "value": repeat_user_scan_share,
            "description": "Share of total scans that came from repeat users.",
            "kind": "percent",
        },
        {
            "label": "Unique users",
            "value": unique_users,
            "description": "Distinct users (AccountId fallback to Email) with at least one scan.",
            "kind": "count",
        },
        {
            "label": "Repeat users",
            "value": repeat_users,
            "description": "Users with more than one scan overall.",
            "kind": "count",
        },
        {
            "label": "Repeat-user share",
            "value": repeat_user_share,
            "description": "Share of users who are repeat users (scanned >1).",
            "kind": "percent",
        },
        {
            "label": "Unique VINs",
            "value": int(vin_counts.size),
            "description": "Distinct VINs scanned at least once.",
            "kind": "count",
        },
        {
            "label": "VINs scanned multiple times",
            "value": int(repeat_vins.sum()),
            "description": "VINs with more than one scan.",
            "kind": "count",
        },
    ]

    state_counts = (
        df.dropna(subset=["State"]).groupby("State").size().sort_values(ascending=False)
    )
    repeat_state_counts = (
        df[df["AccountKey"].isin(repeat_user_ids)]
        .dropna(subset=["State"])
        .groupby("State")
        .size()
    )
    state_summary = pd.DataFrame(
        {
            "state": state_counts.index,
            "total_scans": state_counts.values,
            "repeat_scans": repeat_state_counts.reindex(state_counts.index, fill_value=0)
            .fillna(0)
            .astype(int),
        }
    )
    state_summary["repeat_scan_share"] = state_summary["repeat_scans"].div(
        state_summary["total_scans"]
    )
    state_summary["scan_share"] = state_summary["total_scans"].div(
        total_scans if total_scans else 1
    )

    # Repeat users by state (deduped by user within state).
    user_state = df.dropna(subset=["State"]).drop_duplicates(subset=["State", "AccountKey"])
    repeat_users_by_state = (
        user_state[user_state["AccountKey"].isin(repeat_user_ids)]
        .groupby("State")
        .size()
    )
    unique_users_by_state = user_state.groupby("State").size()
    repeat_user_state = pd.DataFrame({"state": state_counts.index})
    repeat_user_state["repeat_users"] = (
        repeat_users_by_state.reindex(state_counts.index, fill_value=0)
        .astype(int)
        .to_numpy()
    )
    repeat_user_state["repeat_user_share_all_repeat"] = repeat_user_state[
        "repeat_users"
    ].div(repeat_users if repeat_users else 1)
    repeat_user_state["repeat_user_share_in_state"] = (
        repeat_user_state["repeat_users"]
        / unique_users_by_state.reindex(state_counts.index, fill_value=0)
        .replace(0, pd.NA)
    ).fillna(0)

    scans_by_day = (
        df.set_index("CreatedDateTimeUTC")
        .resample("D")
        .size()
        .rename_axis("Date")
        .reset_index(name="Scans")
    )

    return summary, state_summary, repeat_user_state, scans_by_day


def build_figures(
    state_summary: pd.DataFrame,
    repeat_user_state: pd.DataFrame,
    scans_by_day: pd.DataFrame,
) -> Dict[str, go.Figure]:
    ts_fig = px.line(
        scans_by_day,
        x="Date",
        y="Scans",
        title="Daily scans",
        markers=True,
    )

    top_states = state_summary.nlargest(10, "total_scans")
    bar_fig = px.bar(
        top_states,
        x="state",
        y="total_scans",
        hover_data=["repeat_scans", "repeat_scan_share", "scan_share"],
        title="Top states by scans",
        labels={"state": "State", "total_scans": "Scans"},
    )
    bar_fig.update_traces(
        customdata=top_states[
            ["total_scans", "repeat_scans", "repeat_scan_share", "scan_share"]
        ],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Scans: %{customdata[0]:,}<br>"
            "Repeat scans: %{customdata[1]:,}<br>"
            "Repeat scan share: %{customdata[2]:.1%}<br>"
            "Scan share: %{customdata[3]:.1%}<extra></extra>"
        ),
    )

    map_all = px.choropleth(
        state_summary,
        locations="state",
        locationmode="USA-states",
        color="total_scans",
        scope="usa",
        title="Scans by state",
        color_continuous_scale="Blues",
    )
    map_all.update_traces(
        customdata=state_summary[["total_scans", "scan_share"]],
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Total scans: %{customdata[0]:,}<br>"
            "Scan share: %{customdata[1]:.1%}<extra></extra>"
        ),
    )

    repeat_user_map = px.choropleth(
        repeat_user_state,
        locations="state",
        locationmode="USA-states",
        color="repeat_users",
        scope="usa",
        title="Repeat users by state",
        color_continuous_scale="Greens",
    )
    repeat_user_map.update_traces(
        customdata=repeat_user_state[
            [
                "repeat_users",
                "repeat_user_share_all_repeat",
            ]
        ],
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Repeat users: %{customdata[0]:,}<br>"
            "Share of all repeat users: %{customdata[1]:.1%}"
            "<extra></extra>"
        ),
    )

    for fig in (ts_fig, bar_fig, map_all, repeat_user_map):
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))

    return {
        "timeseries": ts_fig,
        "top_states": bar_fig,
        "map_all": map_all,
        "repeat_user_map": repeat_user_map,
    }


def render_html(summary: List[Metric], figures: Dict[str, go.Figure], output_path: Path) -> None:
    """Write a simple, self-contained HTML dashboard."""
    def format_value(metric: Metric) -> str:
        return f"{metric['value']:.1%}" if metric["kind"] == "percent" else f"{metric['value']:,.0f}"

    card_html = "".join(
        f"""
        <div class="card">
            <div class="label">{metric['label']}</div>
            <div class="value">{format_value(metric)}</div>
        </div>
        """
        for metric in summary
    )

    definitions_html = "".join(
        f"<li><strong>{metric['label']}:</strong> {metric['description']}</li>"
        for metric in summary
    )

    ts_html = pio.to_html(
        figures["timeseries"], include_plotlyjs=True, full_html=False
    )
    other_figs = {
        key: pio.to_html(fig, include_plotlyjs=False, full_html=False)
        for key, fig in figures.items()
        if key != "timeseries"
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Oct, 2025 Scan Records Dashboard (RS2 Users)</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f5f6fa; margin: 0; padding: 20px; }}
    h1 {{ margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: white; padding: 14px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .label {{ color: #555; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .value {{ font-size: 22px; font-weight: 600; margin-top: 6px; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .chart {{ background: white; padding: 10px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); min-height: 420px; }}
    .notes {{ background: #eef1f7; border-radius: 8px; padding: 12px 14px; margin-bottom: 18px; color: #333; }}
    .notes ul {{ margin: 8px 0 0 16px; padding: 0; }}
    .notes li {{ margin-bottom: 6px; line-height: 1.4; }}
  </style>
</head>
<body>
  <h1>Oct, 2025 Scan Records Dashboard (RS2 Users)</h1>
  <div class="cards">{card_html}</div>
  <div class="notes">
    <strong>Metric definitions</strong>
    <ul>{definitions_html}</ul>
  </div>
  <div class="charts">
    <div class="chart">{ts_html}</div>
    <div class="chart">{other_figs['top_states']}</div>
    <div class="chart">{other_figs['map_all']}</div>
    <div class="chart">{other_figs['repeat_user_map']}</div>
  </div>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an HTML dashboard from scan_records.csv."
    )
    parser.add_argument(
        "--input",
        default="scan_records.csv",
        help="Path to scans CSV (default: scan_records.csv)",
    )
    parser.add_argument(
        "--output",
        default="index.html",
        help="Output HTML file (default: index.html)",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    output_path = Path(args.output)

    df = load_data(csv_path)
    summary, state_summary, repeat_user_state, scans_by_day = summarize(df)
    figures = build_figures(state_summary, repeat_user_state, scans_by_day)
    render_html(summary, figures, output_path)
    print(f"Dashboard written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
