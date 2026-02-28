# ── components/charts.py ──────────────────────────────────────────────────────
# All Plotly interactive charts. Each function renders directly via st.plotly_chart.

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from config import COLORS, score_color


# ── Safe layout builder — never merges xaxis/yaxis twice ─────────────────────
def _layout(height: int, margin=None, **extra) -> dict:
    """Return a clean Plotly layout dict.
    Pass margin= explicitly to override the default — never duplicate keys."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace",
                  color=COLORS["text_dim"], size=11),
        margin=margin if margin is not None else dict(l=10, r=10, t=10, b=10),
        height=height,
        **extra,
    )


def _axis(**overrides) -> dict:
    """Build an axis dict from the shared defaults + per-call overrides."""
    base = dict(
        gridcolor=COLORS["border"],
        linecolor=COLORS["border"],
        tickcolor=COLORS["border"],
        zerolinecolor=COLORS["border"],
    )
    base.update(overrides)
    return base


def _rgba(hex6: str, alpha: float) -> str:
    """Convert '#RRGGBB' + alpha float to 'rgba(r,g,b,a)' for Plotly."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Radar / Spider Chart ──────────────────────────────────────────────────────
def radar_chart(data: list, key: str = "radar"):
    frameworks = [d["framework"] for d in data]
    scores     = [d["score"]     for d in data]
    fw_loop    = frameworks + [frameworks[0]]
    sc_loop    = scores    + [scores[0]]

    fig = go.Figure(go.Scatterpolar(
        r=sc_loop, theta=fw_loop,
        fill="toself",
        fillcolor=_rgba(COLORS["accent2"], 0.15),
        line=dict(color=COLORS["accent2"], width=2.5),
        marker=dict(color=COLORS["accent2"], size=7,
                    line=dict(color=COLORS["bg"], width=2)),
        hovertemplate="<b>%{theta}</b><br>Score: %{r}%<extra></extra>",
        name="Compliance",
    ))

    fig.update_layout(
        **_layout(300, showlegend=False),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(color=COLORS["muted"], size=9),
                ticksuffix="%",
            ),
            angularaxis=dict(
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(color=COLORS["text_dim"], size=11,
                              family="JetBrains Mono"),
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Area Trend Chart ──────────────────────────────────────────────────────────
def trend_chart(data: list, key: str = "trend"):
    df = pd.DataFrame(data)
    fig = go.Figure(go.Scatter(
        x=df["run"], y=df["score"],
        fill="tozeroy",
        fillcolor=_rgba(COLORS["accent"], 0.08),
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(color=COLORS["accent"], size=8,
                    line=dict(color=COLORS["bg"], width=2)),
        hovertemplate="<b>%{x}</b><br>Score: %{y}%<extra></extra>",
        mode="lines+markers",
        name="Score",
    ))

    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color=_rgba(COLORS["safe"], 0.5),
        line_width=1.5,
        annotation_text="80% target",
        annotation_font=dict(color=COLORS["safe"], size=10),
    )

    fig.update_layout(
        **_layout(260, showlegend=False, hovermode="x unified"),
        xaxis=_axis(showgrid=False),
        yaxis=_axis(range=[0, 108], ticksuffix="%"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Horizontal Bar — Score by scenario ───────────────────────────────────────
def scores_bar_chart(results: list, key: str = "scores_bar"):
    names  = [r["name"][:28] + ("..." if len(r["name"]) > 28 else "") for r in results]
    scores = [r["score"] for r in results]
    colors = [score_color(s) for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=names,
        orientation="h",
        marker=dict(
            color=[_rgba(c, 0.8) for c in colors],
            line=dict(color=colors, width=1),
        ),
        text=[f"{s}%" for s in scores],
        textposition="outside",
        textfont=dict(color=COLORS["text_dim"], size=11, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Score: %{x}%<extra></extra>",
        width=0.6,
    ))

    fig.update_layout(
        **_layout(320, showlegend=False, bargap=0.3),
        xaxis=_axis(range=[0, 115], ticksuffix="%"),
        yaxis=_axis(showgrid=False,
                    tickfont=dict(size=11, family="JetBrains Mono",
                                  color=COLORS["text_dim"])),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Donut — Pass/Fail ratio ───────────────────────────────────────────────────
def donut_chart(results: list, key: str = "donut"):
    compliant  = sum(1 for r in results if r["compliant"])
    violations = len(results) - compliant

    fig = go.Figure(go.Pie(
        labels=["Compliant", "Violations"],
        values=[compliant, violations],
        hole=0.72,
        marker=dict(
            colors=[COLORS["safe"], COLORS["danger"]],
            line=dict(color=COLORS["bg"], width=3),
        ),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        textinfo="none",
    ))

    fig.add_annotation(
        text=f"{compliant}/{len(results)}<br>PASSED",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="JetBrains Mono", color=COLORS["text"], size=16),
    )

    # FIX APPLIED HERE: margin is safely packed inside _layout!
    fig.update_layout(
        **_layout(260, showlegend=True, margin=dict(l=10, r=10, t=10, b=30)),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(color=COLORS["text_dim"], size=12, family="JetBrains Mono"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["border"],
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Gauge — Overall Compliance ────────────────────────────────────────────────
def gauge_chart(score: int, key: str = "gauge"):
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="%",
                    font=dict(size=40, family="JetBrains Mono", color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor=COLORS["border"],
                      tickfont=dict(color=COLORS["text_dim"], size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 50],   color=_rgba(COLORS["danger"], 0.08)),
                dict(range=[50, 80],  color="rgba(255,209,102,0.08)"),
                dict(range=[80, 100], color=_rgba(COLORS["safe"],   0.08)),
            ],
            threshold=dict(line=dict(color=color, width=3),
                           thickness=0.8, value=score),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(**_layout(220, margin=dict(l=20, r=20, t=20, b=10)))
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Heatmap — Severity x Framework ───────────────────────────────────────────
def heatmap_chart(results: list, key: str = "heatmap"):
    frameworks = list({r["framework"] for r in results})
    severities = ["critical", "high", "medium", "low"]

    matrix = []
    for sev in severities:
        row = []
        for fw in frameworks:
            vals = [r["score"] for r in results
                    if r["framework"] == fw and r["severity"] == sev]
            row.append(sum(vals) / len(vals) if vals else None)
        matrix.append(row)

    fig = go.Figure(go.Heatmap(
        z=matrix, x=frameworks, y=severities,
        colorscale=[
            [0.0, COLORS["danger"]],
            [0.5, "#FFD166"],
            [1.0, COLORS["safe"]],
        ],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" if v is not None else "-" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", size=12, color=COLORS["bg"]),
        hovertemplate=(
            "Framework: <b>%{x}</b><br>"
            "Severity: <b>%{y}</b><br>"
            "Avg Score: <b>%{z:.0f}%</b><extra></extra>"
        ),
        colorbar=dict(
            tickfont=dict(color=COLORS["text_dim"], family="JetBrains Mono", size=10),
            ticksuffix="%",
            bgcolor=COLORS["panel"],
            bordercolor=COLORS["border"],
        ),
    ))

    fig.update_layout(
        **_layout(240),
        xaxis=_axis(showgrid=False,
                    tickfont=dict(size=11, family="JetBrains Mono",
                                  color=COLORS["text_dim"])),
        yaxis=_axis(showgrid=False,
                    tickfont=dict(size=11, family="JetBrains Mono",
                                  color=COLORS["text_dim"])),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Waterfall — Score delta by category ──────────────────────────────────────
def waterfall_chart(radar_data: list, key: str = "waterfall"):
    frameworks = [d["framework"] for d in radar_data]
    scores     = [d["score"]     for d in radar_data]
    deltas     = [s - 75 for s in scores]

    fig = go.Figure(go.Waterfall(
        name="vs baseline",
        orientation="v",
        measure=["relative"] * len(deltas),
        x=frameworks,
        y=deltas,
        text=[f"{'+' if d >= 0 else ''}{d}%" for d in deltas],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=COLORS["text_dim"]),
        increasing=dict(marker=dict(
            color=_rgba(COLORS["safe"],   0.73),
            line=dict(color=COLORS["safe"],   width=1),
        )),
        decreasing=dict(marker=dict(
            color=_rgba(COLORS["danger"], 0.73),
            line=dict(color=COLORS["danger"], width=1),
        )),
        connector=dict(line=dict(color=COLORS["border"], width=1, dash="dot")),
        hovertemplate="<b>%{x}</b><br>Delta vs 75%%: %{y}%%<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=_rgba(COLORS["muted"], 0.6),
                  line_width=1, line_dash="dot")

    fig.update_layout(
        **_layout(280, showlegend=False),
        xaxis=_axis(showgrid=False),
        yaxis=_axis(ticksuffix="%"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)