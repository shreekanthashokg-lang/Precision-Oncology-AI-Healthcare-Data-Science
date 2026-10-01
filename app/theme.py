"""
Shared visual theme for the Precision Oncology AI app: custom CSS
(glassmorphism, gradient text, animated background), a reusable hero
banner, styled metric/result cards, and real 3D Plotly visualizations.

Import and call `inject_custom_css()` near the top of every page, right
after `st.set_page_config(...)`.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

PRIMARY = "#14b8a6"    # teal
SECONDARY = "#8b5cf6"  # violet
ACCENT = "#f472b6"     # pink
BG_DARK = "#0b1220"
CARD_BG = "rgba(255, 255, 255, 0.04)"


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        h1, h2, h3, .hero-title {{
            font-family: 'Sora', sans-serif !important;
        }}

        /* App background: deep-space gradient with a soft animated glow */
        .stApp {{
            background: radial-gradient(circle at 15% 10%, #16213d 0%, {BG_DARK} 45%, #070b14 100%);
            background-attachment: fixed;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0d1526 0%, #0a0f1c 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        section[data-testid="stSidebar"] * {{
            color: #cbd5e1 !important;
        }}

        /* Hero banner */
        .hero-wrap {{
            padding: 2.2rem 2.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(20,184,166,0.14) 0%, rgba(139,92,246,0.14) 55%, rgba(244,114,182,0.10) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
            margin-bottom: 1.6rem;
        }}
        .hero-title {{
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, {PRIMARY}, {SECONDARY} 55%, {ACCENT});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .hero-sub {{
            color: #a8b3c7;
            font-size: 1.02rem;
            margin-top: 0.5rem;
            max-width: 900px;
        }}

        /* Glass cards */
        .glass-card {{
            background: {CARD_BG};
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 28px rgba(20,184,166,0.15);
            border-color: rgba(20,184,166,0.35);
        }}
        .glass-metric-label {{
            color: #8a94a6;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .glass-metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-top: 0.15rem;
        }}
        .glass-metric-delta {{
            font-size: 0.85rem;
            color: {PRIMARY};
            margin-top: 0.2rem;
        }}

        /* Result banners */
        .result-good {{
            background: linear-gradient(135deg, rgba(20,184,166,0.16), rgba(20,184,166,0.05));
            border: 1px solid rgba(20,184,166,0.4);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            color: #e6fffb;
        }}
        .result-warn {{
            background: linear-gradient(135deg, rgba(244,114,182,0.18), rgba(244,114,182,0.05));
            border: 1px solid rgba(244,114,182,0.4);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            color: #ffe9f4;
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(139,92,246,0.35);
        }}

        /* Sliders / selects accent */
        div[data-baseweb="slider"] > div > div > div {{
            background: linear-gradient(90deg, {PRIMARY}, {SECONDARY}) !important;
        }}

        /* Chip/badge for nav-like labels */
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background: rgba(20,184,166,0.15);
            border: 1px solid rgba(20,184,166,0.4);
            color: {PRIMARY};
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }}

        /* Tighten default Streamlit top padding */
        .block-container {{
            padding-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-title">{icon} {title}</div>
            <div class="hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(items) -> None:
    """items: list of (label, value, delta_or_None)"""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        delta_html = f'<div class="glass-metric-delta">{delta}</div>' if delta else ""
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="glass-metric-label">{label}</div>
                    <div class="glass-metric-value">{value}</div>
                    {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_result_banner(kind: str, text: str) -> None:
    css_class = "result-good" if kind == "good" else "result-warn"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def render_dna_helix_3d(height: int = 460) -> go.Figure:
    """
    A real, mathematically-generated 3D double-helix (two offset sine/cosine
    strands + connecting base-pair rungs), fully interactive — drag to
    rotate, scroll to zoom. Purely a visual centerpiece for the Home page.
    """
    t = np.linspace(0, 4 * np.pi, 160)
    x1, y1, z1 = np.cos(t), np.sin(t), t
    x2, y2, z2 = np.cos(t + np.pi), np.sin(t + np.pi), t

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1, mode="lines",
        line=dict(color=PRIMARY, width=8), name="Strand A", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2, mode="lines",
        line=dict(color=SECONDARY, width=8), name="Strand B", hoverinfo="skip",
    ))
    # Base-pair rungs every few steps, colored like a real double helix rendering
    rung_colors = [ACCENT, "#60a5fa", "#facc15", "#34d399"]
    for i in range(0, len(t), 6):
        fig.add_trace(go.Scatter3d(
            x=[x1[i], x2[i]], y=[y1[i], y2[i]], z=[z1[i], z2[i]],
            mode="lines",
            line=dict(color=rung_colors[i % len(rung_colors)], width=3),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.6)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
