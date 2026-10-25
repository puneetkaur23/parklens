"""
ui/charts.py – High-contrast interactive Plotly visualization helpers.

Themes:
  - Dark glassmorphic transparency
  - Electric Teal (#22D3EE), Emerald Green (#22C55E), Crimson Red (#EF4444), Amber (#F59E0B)
  - Clear typography with tabular figure rendering
  - Donut chart with centered total capacity and occupancy breakdown
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import pandas as pd


def create_occupancy_donut(free_count: int, occupied_count: int) -> go.Figure:
    """
    Render interactive donut chart of parking slot occupancy with center summary badge.
    Colors: Green (#22C55E) for Free, Red (#EF4444) for Occupied.
    """
    total = free_count + occupied_count
    labels = ["Free Spaces", "Occupied Vehicles"]
    values = [free_count, occupied_count]
    colors = ["#22C55E", "#EF4444"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.68,
        marker=dict(colors=colors, line=dict(color="#0B1118", width=3)),
        textinfo="percent",
        textfont=dict(color="#FFFFFF", size=12, family="Inter, sans-serif"),
        hoverinfo="label+value+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value} spaces<br>Share: %{percent}<extra></extra>"
    )])

    occ_rate = (occupied_count / total * 100) if total > 0 else 0
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color="#CBD5E1", size=11)
        ),
        margin=dict(l=10, r=10, t=10, b=30),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b style='font-size:24px;color:#F8FAFC;'>{occ_rate:.0f}%</b><br><span style='font-size:11px;color:#94A3B8;'>{occupied_count}/{total} OCCUPIED</span>",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(family="Inter, sans-serif")
        )]
    )
    return fig


def create_feature_bars(features: Dict[str, float]) -> go.Figure:
    """Render horizontal bar chart of extracted classical features for Space Inspector."""
    names = ["Edge Density", "Diff Score", "Laplacian Var (norm)", "Luminance Std (norm)"]
    vals = [
        features.get("edge_density", 0.0),
        features.get("diff_score", 0.0),
        min(1.0, features.get("laplacian_variance", 0.0) / 1500.0),
        min(1.0, features.get("gray_std", 0.0) / 80.0)
    ]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0, "#22D3EE"], [0.5, "#22C55E"], [1, "#F59E0B"]],
            cmin=0.0, cmax=1.0,
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        ),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#CBD5E1", size=11, family="Inter, monospace")
    ))

    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        height=180,
        xaxis=dict(
            range=[0, 1.15],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            tickfont=dict(color="#94A3B8", size=10)
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#E2E8F0", size=11)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif")
    )
    return fig


def create_timeline_chart(timeline_records: List[Dict[str, Any]]) -> go.Figure:
    """Render interactive line chart of diurnal occupancy fluctuations with hover tooltips."""
    df = pd.DataFrame(timeline_records)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_label"],
        y=df["occupied"],
        mode="lines+markers",
        name="Occupied Spaces",
        line=dict(color="#EF4444", width=3, shape="spline"),
        marker=dict(size=7, color="#EF4444", line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="<b>Time:</b> %{x}<br><b>Occupied:</b> %{y} spaces<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df["time_label"],
        y=df["free"],
        mode="lines+markers",
        name="Available Spaces",
        line=dict(color="#22C55E", width=3, shape="spline"),
        marker=dict(size=7, color="#22C55E", line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="<b>Time:</b> %{x}<br><b>Available:</b> %{y} spaces<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Diurnal Parking Occupancy Curve",
            font=dict(size=14, color="#F8FAFC", family="Inter, sans-serif")
        ),
        xaxis_title="Time of Day (Timestamp)",
        yaxis_title="Spaces Count",
        margin=dict(l=30, r=20, t=45, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11, 17, 24, 0.75)",
        font=dict(color="#94A3B8", family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94A3B8")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1", size=11)
        )
    )
    return fig
