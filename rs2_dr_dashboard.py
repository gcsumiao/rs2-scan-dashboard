#!/usr/bin/env python3
"""Generate an HTML dashboard for rs2_dr_0901.csv."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, TypedDict

import json
import pandas as pd


class Metric(TypedDict):
    label: str
    value: float
    description: str
    kind: str  # "count", "percent", or "float"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and normalize the dataset."""
    df = pd.read_csv(csv_path)
    df["CreatedDateTimeUTC"] = pd.to_datetime(
        df["CreatedDateTimeUTC"], errors="coerce", utc=True
    )
    df["CreatedDateTimePT"] = df["CreatedDateTimeUTC"].dt.tz_convert("US/Pacific")
    df["State"] = df["State"].astype(str).str.strip().str.upper()
    df.loc[~df["State"].str.fullmatch(r"[A-Z]{2}"), "State"] = pd.NA
    for col in ["Year", "Mileage", "TotalAbsCodes", "TotalSrsCodes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stable user identifier: AccountId only for user-based metrics.
    df["UserId"] = df["AccountId"].astype(str).str.strip()
    df.loc[df["UserId"].isin(["", "nan"]), "UserId"] = pd.NA

    df["VehicleAge"] = 2026 - df["Year"]
    df.loc[(df["Year"] < 1980) | (df["Year"] > 2026), "VehicleAge"] = pd.NA

    def vehicle_label(row: pd.Series) -> str | None:
        if pd.isna(row["Year"]) or pd.isna(row["Make"]) or pd.isna(row["Model"]):
            return None
        try:
            year = int(row["Year"])
        except (TypeError, ValueError):
            return None
        make = str(row["Make"]).strip()
        model = str(row["Model"]).strip()
        if not make or not model:
            return None
        return f"{year} {make} {model}"

    df["Vehicle"] = df.apply(vehicle_label, axis=1)
    return df


def explode_parts(series: pd.Series) -> Counter:
    """Split semicolon-separated part names and count occurrences."""
    counter: Counter = Counter()
    for raw in series.dropna():
        for part in str(raw).split(";"):
            name = part.strip()
            if name:
                counter[name] += 1
    return counter


def top_mil_dtc(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    dtc_counts = df["MIL DTC"].dropna().astype(str).str.strip().value_counts()
    rows = []
    for dtc, count in dtc_counts.head(limit).items():
        parts_counter = explode_parts(
            df.loc[df["MIL DTC"].astype(str).str.strip() == dtc, "MIL Part Name"]
        )
        if parts_counter:
            parts_str = ", ".join(
                f"{name} ({cnt})" for name, cnt in parts_counter.most_common(3)
            )
        else:
            parts_str = "n/a"
        rows.append({"MIL DTC": dtc, "Scans": int(count), "Top Parts": parts_str})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    total_scans = len(df)
    def safe_mean(series: pd.Series) -> float:
        cleaned = series.dropna()
        return float(cleaned.mean()) if not cleaned.empty else 0.0

    avg_age = safe_mean(df["VehicleAge"])
    avg_mileage = safe_mean(df["Mileage"])
    avg_abs_codes = safe_mean(df["TotalAbsCodes"])
    avg_srs_codes = safe_mean(df["TotalSrsCodes"])
    maintenance_parts_total = int(df["MaintenancePartsCount"].sum())
    predicted_parts_total = int(df["PredictedPartsCount"].sum())
    unique_vins = int(df["VIN"].dropna().nunique())

    user_counts = df["UserId"].dropna().value_counts()
    total_scanned_users = int(user_counts.size)

    # Repeat users: AccountId that scanned the same VIN 2+ times.
    repeat_pairs = (
        df.dropna(subset=["UserId", "VIN"])
        .groupby(["UserId", "VIN"])
        .size()
        .reset_index(name="cnt")
    )
    repeat_accounts = repeat_pairs[repeat_pairs["cnt"] >= 2]["UserId"].unique()
    repeat_user_count = int(len(repeat_accounts))
    repeat_user_share = repeat_user_count / total_scanned_users if total_scanned_users else 0

    cards: List[Metric] = [
        {
            "label": "Total scans",
            "value": total_scans,
            "description": "All scan events in the dataset.",
            "kind": "count",
        },
        {
            "label": "Total scanned users",
            "value": total_scanned_users,
            "description": "Distinct AccountId values with at least one scan.",
            "kind": "count",
        },
        {
            "label": "Average vehicle age",
            "value": avg_age,
            "description": "Mean of (2026 - Year) for valid years.",
            "kind": "float",
        },
        {
            "label": "Average mileage",
            "value": avg_mileage,
            "description": "Mean of mileage across scans.",
            "kind": "float",
        },
        {
            "label": "Average ABS code count",
            "value": avg_abs_codes,
            "description": "Mean of TotalAbsCodes where present.",
            "kind": "float",
        },
        {
            "label": "Average SRS code count",
            "value": avg_srs_codes,
            "description": "Mean of TotalSrsCodes where present.",
            "kind": "float",
        },
        {
            "label": "Maintenance parts (total)",
            "value": maintenance_parts_total,
            "description": "Sum of MaintenancePartsCount.",
            "kind": "count",
        },
        {
            "label": "Predicted parts (total)",
            "value": predicted_parts_total,
            "description": "Sum of PredictedPartsCount.",
            "kind": "count",
        },
        {
            "label": "Unique VINs",
            "value": unique_vins,
            "description": "Distinct VINs scanned.",
            "kind": "count",
        },
        {
            "label": "Repeat users",
            "value": repeat_user_count,
            "description": "AccountId values that scanned the same VIN at least twice.",
            "kind": "count",
        },
        {
            "label": "Repeat user share",
            "value": repeat_user_share,
            "description": "Share of scanned users who repeated on the same VIN.",
            "kind": "percent",
        },
    ]

    top_vehicles = (
        df["Vehicle"].dropna().value_counts().head(10).rename_axis("Vehicle").reset_index(name="Scans")
    )
    top_usb = (
        df["UsbProductId"]
        .dropna()
        .astype(int)
        .value_counts()
        .head(5)
        .rename_axis("UsbProductId")
        .reset_index(name="Scans")
    )

    abs_parts_counter = explode_parts(df["ABS Part Name"])
    abs_parts = (
        pd.DataFrame(abs_parts_counter.most_common(10), columns=["ABS Part Name", "Mentions"])
        if abs_parts_counter
        else pd.DataFrame(columns=["ABS Part Name", "Mentions"])
    )

    srs_parts_counter = explode_parts(df["SRS Part Name"])
    srs_parts = (
        pd.DataFrame(srs_parts_counter.most_common(10), columns=["SRS Part Name", "Mentions"])
        if srs_parts_counter
        else pd.DataFrame(columns=["SRS Part Name", "Mentions"])
    )

    mil_dtc_df = top_mil_dtc(df, limit=10)

    state_counts = (
        df.dropna(subset=["State"]).groupby("State").size().sort_values(ascending=False)
    )
    scans_by_state = state_counts.rename("Scans").reset_index()

    if df["CreatedDateTimeUTC"].notna().any():
        hourly = (
            df.set_index("CreatedDateTimeUTC")
            .resample("h")
            .size()
            .rename_axis("Hour")
            .reset_index(name="Scans")
        )
        if not hourly.empty:
            max_time = hourly["Hour"].max()
            window_start = max_time - pd.Timedelta(hours=23)
            hourly = hourly[hourly["Hour"] >= window_start]
    else:
        hourly = pd.DataFrame(columns=["Hour", "Scans"])

    repeat_users_by_state = (
        df[df["UserId"].isin(repeat_accounts)]
        .dropna(subset=["State"])
        .groupby("State")["UserId"]
        .nunique()
        .rename("RepeatUsers")
        .reset_index()
    )

    def to_records(df: pd.DataFrame) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for row in df.to_dict("records"):
            for key, val in list(row.items()):
                if pd.isna(val):
                    row[key] = None
                elif isinstance(val, pd.Timestamp):
                    row[key] = val.isoformat()
            records.append(row)
        return records

    chart_data = {
        "hourly": to_records(hourly) if isinstance(hourly, pd.DataFrame) else [],
        "scans_by_state": to_records(scans_by_state),
        "repeat_users_by_state": to_records(repeat_users_by_state),
        "top_states": to_records(scans_by_state.head(10)),
        "top_vehicles": to_records(top_vehicles),
        "top_usb": to_records(top_usb),
        "mil_dtc": to_records(mil_dtc_df),
        "abs_parts": to_records(abs_parts),
        "srs_parts": to_records(srs_parts),
    }

    return {
        "cards": cards,
        "top_vehicles": top_vehicles,
        "top_usb": top_usb,
        "mil_dtc": mil_dtc_df,
        "abs_parts": abs_parts,
        "srs_parts": srs_parts,
        "scans_by_state": scans_by_state,
        "hourly": hourly,
        "repeat_users_by_state": repeat_users_by_state,
        "chart_data": chart_data,
    }


def render_html(
    cards: List[Metric],
    tables: Dict[str, pd.DataFrame],
    chart_data: Dict[str, object],
    output_path: Path,
) -> None:
    def format_value(metric: Metric) -> str:
        if metric["kind"] == "percent":
            return f"{metric['value']:.1%}"
        if metric["kind"] == "float":
            return f"{metric['value']:,.2f}"
        return f"{metric['value']:,.0f}"

    card_html = "".join(
        f"""
        <div class="card">
            <div class="label">{m['label']}</div>
            <div class="value">{format_value(m)}</div>
        </div>
        """
        for m in cards
    )

    definitions_html = "".join(
        f"<li><strong>{m['label']}:</strong> {m['description']}</li>" for m in cards
    )

    table_sections = []
    table_titles = {
        "top_vehicles": "Top 10 vehicles (Year, Make, Model)",
        "top_usb": "Top 5 scan tools (UsbProductId)",
        "mil_dtc": "Top 10 MIL DTC with common parts",
        "abs_parts": "Top 10 ABS part names",
        "srs_parts": "Top 10 SRS part names",
    }
    for key, title in table_titles.items():
        df = tables.get(key, pd.DataFrame())
        has_data = isinstance(df, pd.DataFrame) and not df.empty
        if has_data:
            table_html = df.to_html(index=False, classes="data-table")
        else:
            table_html = "<p class='muted'>No data available.</p>"
        button = (
            f"<button class='btn' onclick=\"downloadTable('{key}')\">Download CSV</button>"
            if has_data
            else "<button class='btn' disabled>Download CSV</button>"
        )
        section = (
            f"<div class='table-block'>"
            f"<div class='table-head'><h3>{title}</h3>{button}</div>"
            f"{table_html}</div>"
        )
        table_sections.append(section)

    chart_json = json.dumps(chart_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>rs2 scan 2025-09-01</title>
  <style>
    :root {{
      --bg: #0f172a;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --card-bg: #0b1221;
      --card-border: #1f2937;
      --plot-bg: #0b1221;
      --grid: #1f2937;
      --accent-1: #38bdf8;
      --accent-2: #fbbf24;
      --accent-3: #34d399;
    }}
    body.light {{
      --bg: #f6f8fb;
      --text: #0f172a;
      --muted: #4b5563;
      --card-bg: #ffffff;
      --card-border: #e5e7eb;
      --plot-bg: #ffffff;
      --grid: #e5e7eb;
      --accent-1: #2563eb;
      --accent-2: #ea580c;
      --accent-3: #16a34a;
    }}
    body {{
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
      transition: background 0.2s ease, color 0.2s ease;
    }}
    h1 {{ margin: 0; }}
    h3 {{ margin-bottom: 6px; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .toggle {{
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      color: #fff;
      border: none;
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
      font-weight: 600;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    .btn {{
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      color: #fff;
      border: none;
      border-radius: 999px;
      padding: 6px 12px;
      cursor: pointer;
      font-weight: 600;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .btn:disabled {{
      opacity: 0.55;
      cursor: not-allowed;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 12px 14px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      margin: 18px 0;
    }}
    .chart {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 10px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }}
    .tables {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    .table-block {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 10px 12px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }}
    .table-head, .chart-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      color: var(--text);
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      border: 1px solid var(--card-border);
      padding: 6px 8px;
      text-align: left;
    }}
    .data-table th {{ background: var(--plot-bg); }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .notes {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 18px;
    }}
    .notes ul {{ margin: 6px 0 0 16px; padding: 0; }}
    .notes li {{ margin-bottom: 4px; line-height: 1.5; }}
  </style>
</head>
<body class="dark">
  <div class="header">
    <h1>rs2 scan 2025-09-01</h1>
    <button id="theme-toggle" class="toggle" type="button">Switch to light mode</button>
  </div>
  <div class="cards">{card_html}</div>
  <div class="notes">
    <strong>Metric definitions</strong>
    <ul>{definitions_html}</ul>
  </div>
  <div class="tables">
    {''.join(table_sections)}
  </div>
  <div class="charts">
    <div class="chart">
      <div class="chart-head">
        <span>Scans in the last 24 hours (UTC)</span>
        <button class="btn" onclick="downloadChart('hourly-chart','hourly_scans')">Download PNG</button>
      </div>
      <div id="hourly-chart"></div>
    </div>
    <div class="chart">
      <div class="chart-head">
        <span>Scans by state</span>
        <button class="btn" onclick="downloadChart('state-map','state_map')">Download PNG</button>
      </div>
      <div id="state-map"></div>
    </div>
    <div class="chart">
      <div class="chart-head">
        <span>Repeat users by state</span>
        <button class="btn" onclick="downloadChart('repeat-map','repeat_user_map')">Download PNG</button>
      </div>
      <div id="repeat-map"></div>
    </div>
    <div class="chart">
      <div class="chart-head">
        <span>Top states by scans</span>
        <button class="btn" onclick="downloadChart('state-bar','top_states')">Download PNG</button>
      </div>
      <div id="state-bar"></div>
    </div>
    <div class="chart">
      <div class="chart-head">
        <span>Top vehicles by scans</span>
        <button class="btn" onclick="downloadChart('vehicle-bar','top_vehicles')">Download PNG</button>
      </div>
      <div id="vehicle-bar"></div>
    </div>
  </div>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script>
    const chartData = {chart_json};

    function themeColors() {{
      const styles = getComputedStyle(document.body);
      return {{
        bg: styles.getPropertyValue('--card-bg').trim(),
        plot: styles.getPropertyValue('--plot-bg').trim(),
        text: styles.getPropertyValue('--text').trim(),
        grid: styles.getPropertyValue('--grid').trim(),
        accent1: styles.getPropertyValue('--accent-1').trim(),
        accent2: styles.getPropertyValue('--accent-2').trim(),
        accent3: styles.getPropertyValue('--accent-3').trim(),
      }};
    }}

    function renderHourly(colors) {{
      const data = chartData.hourly || [];
      if (!data.length) return;
      const x = data.map(d => d.Hour);
      const y = data.map(d => d.Scans);
      Plotly.react('hourly-chart', [{{
        x, y, type: 'scatter', mode: 'lines+markers', line: {{color: colors.accent1}}
      }}], {{
        title: 'Scans in the last 24 hours (UTC, by hour)',
        paper_bgcolor: colors.bg,
        plot_bgcolor: colors.plot,
        font: {{color: colors.text}},
        xaxis: {{gridcolor: colors.grid}},
        yaxis: {{gridcolor: colors.grid}},
        margin: {{t: 40, l: 40, r: 20, b: 60}}
      }});
    }}

    function renderStateMap(colors) {{
      const data = chartData.scans_by_state || [];
      if (!data.length) return;
      Plotly.react('state-map', [{{
        type: 'choropleth',
        locationmode: 'USA-states',
        locations: data.map(d => d.State),
        z: data.map(d => d.Scans),
        colorscale: [
          [0, '#e0f2fe'],
          [1, '#1d4ed8']
        ],
        colorbar: {{title: 'Scans'}}
      }}], {{
        title: 'Scans by state',
        paper_bgcolor: colors.bg,
        plot_bgcolor: colors.plot,
        font: {{color: colors.text}},
        geo: {{scope: 'usa', bgcolor: colors.plot}},
        margin: {{t: 40, l: 20, r: 20, b: 20}}
      }});
    }}

    function renderRepeatMap(colors) {{
      const data = chartData.repeat_users_by_state || [];
      if (!data.length) return;
      Plotly.react('repeat-map', [{{
        type: 'choropleth',
        locationmode: 'USA-states',
        locations: data.map(d => d.State),
        z: data.map(d => d.RepeatUsers),
        colorscale: [
          [0, '#dcfce7'],
          [1, '#166534']
        ],
        colorbar: {{title: 'Repeat users'}}
      }}], {{
        title: 'Repeat users by state',
        paper_bgcolor: colors.bg,
        plot_bgcolor: colors.plot,
        font: {{color: colors.text}},
        geo: {{scope: 'usa', bgcolor: colors.plot}},
        margin: {{t: 40, l: 20, r: 20, b: 20}}
      }});
    }}

    function renderStateBar(colors) {{
      const data = chartData.top_states || [];
      if (!data.length) return;
      Plotly.react('state-bar', [{{
        type: 'bar',
        x: data.map(d => d.State),
        y: data.map(d => d.Scans),
        marker: {{color: colors.accent1}}
      }}], {{
        title: 'Top states by scans',
        paper_bgcolor: colors.bg,
        plot_bgcolor: colors.plot,
        font: {{color: colors.text}},
        xaxis: {{gridcolor: colors.grid}},
        yaxis: {{gridcolor: colors.grid}},
        margin: {{t: 40, l: 40, r: 20, b: 80}}
      }});
    }}

    function renderVehicleBar(colors) {{
      const data = chartData.top_vehicles || [];
      if (!data.length) return;
      Plotly.react('vehicle-bar', [{{
        type: 'bar',
        x: data.map(d => d.Vehicle),
        y: data.map(d => d.Scans),
        marker: {{color: colors.accent2}}
      }}], {{
        title: 'Top vehicles by scans',
        paper_bgcolor: colors.bg,
        plot_bgcolor: colors.plot,
        font: {{color: colors.text}},
        xaxis: {{tickangle: -40, gridcolor: colors.grid}},
        yaxis: {{gridcolor: colors.grid}},
        margin: {{t: 40, l: 40, r: 20, b: 120}}
      }});
    }}

    function renderCharts() {{
      const colors = themeColors();
      renderHourly(colors);
      renderStateMap(colors);
      renderRepeatMap(colors);
      renderStateBar(colors);
      renderVehicleBar(colors);
    }}

    const toggle = document.getElementById('theme-toggle');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('rs2Theme');
    const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');

    function setTheme(theme) {{
      document.body.classList.toggle('dark', theme === 'dark');
      document.body.classList.toggle('light', theme === 'light');
      toggle.textContent = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
      localStorage.setItem('rs2Theme', theme);
      renderCharts();
    }}

    function downloadChart(id, filename) {{
      const el = document.getElementById(id);
      if (!el) return;
      Plotly.downloadImage(el, {{
        format: 'png',
        filename,
        height: 600,
        width: 900,
      }});
    }}

    function toCsv(rows) {{
      if (!rows || !rows.length) return '';
      const headers = Object.keys(rows[0]);
      const escape = (val) => {{
        if (val === null || val === undefined) return '';
        const s = String(val);
        if (/[\",\\n]/.test(s)) return '\"' + s.replace(/\"/g, '\"\"') + '\"';
        return s;
      }};
      const lines = [headers.join(',')];
      rows.forEach((row) => {{
        lines.push(headers.map(h => escape(row[h])).join(','));
      }});
      return lines.join('\\n');
    }}

    function downloadTable(key) {{
      const data = chartData[key];
      if (!data || !data.length) return;
      const csv = toCsv(data);
      const blob = new Blob([csv], {{type: 'text/csv;charset=utf-8;'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${{key}}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    toggle.addEventListener('click', () => {{
      const next = document.body.classList.contains('dark') ? 'light' : 'dark';
      setTheme(next);
    }});

    setTheme(initialTheme);
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard for rs2_dr_0901.csv")
    parser.add_argument(
        "--input",
        default="rs2_dr_0901.csv",
        help="Path to input CSV (default: rs2_dr_0901.csv)",
    )
    parser.add_argument(
        "--output",
        default="rs2_dashboard.html",
        help="Path to output HTML (default: rs2_dashboard.html)",
    )
    args = parser.parse_args()

    df = load_data(Path(args.input))
    summary = summarize(df)
    render_html(
        cards=summary["cards"],
        tables={
            "top_vehicles": summary["top_vehicles"],
            "top_usb": summary["top_usb"],
            "mil_dtc": summary["mil_dtc"],
            "abs_parts": summary["abs_parts"],
            "srs_parts": summary["srs_parts"],
        },
        chart_data=summary["chart_data"],
        output_path=Path(args.output),
    )
    print(f"Dashboard written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
