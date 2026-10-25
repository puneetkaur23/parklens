"""
ui/components.py - Reusable Presentation Components & Design System Primitives.

Features:
  - Base64 offline background image caching & CSS injection (bg1, bg2, bg3)
  - Hero Header with compact branded logo and live data badge
  - Modern Glassmorphism KPI metric cards with tabular figures
  - Realistic Parking Bay Grid with pulsating free spots and vehicle outlines (no emojis)
  - Pipeline Filmstrip with numbered steps and arrow connectors
  - Breadcrumb context bar, status strip, and dismissible Quick Start guide
  - Empty state cards for zero-data conditions
"""

import cv2
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource(show_spinner=False)
def get_background_data_uris() -> Dict[str, str]:
    """
    Load and downscale background images (bg1, bg2, bg3) once,
    returning base64 data URIs for offline, fast full-page presentation.
    """
    bg_dir = PROJECT_ROOT / "assets" / "backgrounds"
    uris = {}

    for name in ["bg1.jpeg", "bg2.jpeg", "bg3.jpeg"]:
        p = bg_dir / name
        if not p.exists():
            uris[name] = ""
            continue

        img = cv2.imread(str(p))
        if img is None:
            uris[name] = ""
            continue

        # Downscale if wider than 1600px to maintain peak responsiveness
        h, w = img.shape[:2]
        if w > 1600:
            scale = 1600.0 / w
            img = cv2.resize(img, (1600, int(h * scale)), interpolation=cv2.INTER_AREA)

        # JPEG encode at quality 75
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        b64 = base64.b64encode(buf).decode("utf-8")
        uris[name] = f"data:image/jpeg;base64,{b64}"

    return uris


def inject_custom_styles() -> None:
    """Read styles.css, bind background data URIs, and inject into Streamlit DOM."""
    css_path = PROJECT_ROOT / "ui" / "styles.css"
    if not css_path.exists():
        return

    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()

    bgs = get_background_data_uris()
    bg1_uri = bgs.get("bg1.jpeg", "")
    bg2_uri = bgs.get("bg2.jpeg", "")
    bg3_uri = bgs.get("bg3.jpeg", "")

    dynamic_bg_css = f"""
    <style>
      {css_content}

      /* Dynamic background assignments with lightened overlay & vignette */
      .stApp {{
        background-image: linear-gradient(180deg, rgba(5, 8, 12, 0.76) 0%, rgba(5, 8, 12, 0.83) 100%),
                          radial-gradient(circle at 50% 25%, rgba(34, 211, 238, 0.04) 0%, rgba(5, 8, 12, 0.35) 100%),
                          url('{bg1_uri}') !important;
      }}

      .hero-header-banner {{
        background-image: url('{bg2_uri}') !important;
      }}

      .about-backdrop-card {{
        background-image: url('{bg3_uri}') !important;
      }}
    </style>
    """
    st.markdown(dynamic_bg_css, unsafe_allow_html=True)


def render_hero_header(is_synthetic: bool = False) -> None:
    """
    Render compact top product hero banner with parking lens icon, wordmark,
    tagline, and data mode badge. No placeholder or unverifiable meta rows.
    """
    badge_html = (
        """<div class="badge-pill badge-demo"><span class="badge-dot amber"></span> DEMO DATA</div>"""
        if is_synthetic else
        """<div class="badge-pill badge-real"><span class="pulse-dot"></span> PKLot (Real Data)</div>"""
    )

    st.markdown(f"""
    <div class="hero-header-banner">
      <div class="hero-overlay"></div>
      <div class="hero-content-left">
        <div class="hero-title-group">
          <div class="hero-logo-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="2" y="2" width="20" height="20" rx="5" stroke="#05080C" stroke-width="2" fill="#22D3EE"/>
              <path d="M9 17V7H13.5C15.433 7 17 8.567 17 10.5C17 12.433 15.433 14 13.5 14H9" stroke="#05080C" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <div>
            <h1 class="hero-wordmark">Park<span>Lens</span></h1>
            <div class="hero-tagline">Classical Computer Vision Parking Occupancy Detector</div>
          </div>
        </div>
      </div>
      <div class="hero-content-right">
        {badge_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_context_bar(section: str, lot: str, weather: str, filename: str) -> None:
    """Render slim context breadcrumb bar under the tabs."""
    st.markdown(f"""
    <div class="context-bar">
      <span class="ctx-crumb">{section}</span>
      <span class="ctx-sep">/</span>
      <span class="ctx-crumb">{lot}</span>
      <span class="ctx-sep">/</span>
      <span class="ctx-crumb">{weather}</span>
      <span class="ctx-sep">/</span>
      <span class="ctx-leaf">{filename}</span>
    </div>
    """, unsafe_allow_html=True)


def render_status_strip(is_synthetic: bool, space_count: int, method_desc: str = "Homography + CLAHE + Canny + Diff") -> None:
    """Render compact status strip showing data provenance and monitored spaces."""
    badge_label = "Demo Data" if is_synthetic else "PKLot (Real Data)"
    dot_color = "amber" if is_synthetic else "green"
    st.markdown(f"""
    <div class="status-strip">
      <div class="status-left">
        <span class="status-pill"><span class="badge-dot {dot_color}"></span> {badge_label}</span>
        <span class="status-item"><b>{space_count}</b> spaces monitored</span>
      </div>
      <div class="status-right">
        <span class="status-method">Pipeline: {method_desc}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_page_header(title: str, description: str) -> None:
    """Render short header with plain-English summary for each tab."""
    st.markdown(f"""
    <div class="tab-page-header">
      <h3 class="tab-page-title">{title}</h3>
      <p class="tab-page-desc">{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_how_to_use_banner() -> None:
    """Render first-visit guidance banner on Live Detection tab with dismissal."""
    if st.session_state.get("dismiss_how_to_use", False):
        return

    st.markdown("""
    <div class="how-to-card">
      <div class="how-to-header">
        <div class="how-to-title">Quick Start Guide</div>
      </div>
      <div class="how-to-steps">
        <div class="how-step">
          <div class="how-num">1</div>
          <div class="how-body">
            <div class="how-step-title">Select Parking Lot & Weather</div>
            <div class="how-step-desc">Use the sidebar Quick Demo presets or pick a specific frame from the PKLot dataset.</div>
          </div>
        </div>
        <div class="how-step">
          <div class="how-num">2</div>
          <div class="how-body">
            <div class="how-step-title">Inspect Detection Overlay</div>
            <div class="how-step-desc">Green polygons indicate free spaces; red polygons denote occupied vehicles.</div>
          </div>
        </div>
        <div class="how-step">
          <div class="how-num">3</div>
          <div class="how-body">
            <div class="how-step-title">Analyze & Export</div>
            <div class="how-step-desc">Inspect per-space features in Pipeline Explorer or export the annotated frame and CSV.</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_space, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Dismiss Guide", key="dismiss_how_btn", use_container_width=True):
            st.session_state["dismiss_how_to_use"] = True
            st.rerun()


def render_kpi_card(title: str, value: str, color_class: str = "teal", subtitle: str = "", tooltip: str = "") -> None:
    """Render modern glassmorphic KPI card with responsive tabular numbers."""
    sub_html = f'<div class="kpi-subtext">{subtitle}</div>' if subtitle else ""
    title_attr = f'title="{tooltip}"' if tooltip else (f'title="{subtitle}"' if subtitle else "")
    st.markdown(f"""
    <div class="kpi-card-v2" {title_attr}>
      <div class="kpi-accent-bar {color_class}"></div>
      <div class="kpi-header">
        <span class="kpi-label">{title}</span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-number {color_class}">{value}</span>
      </div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_space_legend(highlight_errors: bool = True) -> None:
    """Render small legend directly beneath the annotated image."""
    err_html = """
    <span class="legend-chip"><span class="legend-box yellow"></span> Misclassified</span>
    """ if highlight_errors else ""

    st.markdown(f"""
    <div class="space-legend-bar">
      <span class="legend-chip"><span class="legend-box green"></span> Free Space</span>
      <span class="legend-chip"><span class="legend-box red"></span> Occupied Space</span>
      {err_html}
    </div>
    """, unsafe_allow_html=True)


def render_parking_bays(results: List[Dict[str, Any]]) -> None:
    """Render top-down realistic parking stall bay grid with free pulse and vehicle outlines (no emojis)."""
    bays_html = []
    
    # Clean SVG icons
    svg_car = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 17h14M5 17a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1.5l1.8-3.6A2 2 0 0 1 10.1 2h3.8a2 2 0 0 1 1.8 1.4L17.5 7H19a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2M5 17a2 2 0 1 0 4 0M15 17a2 2 0 1 0 4 0"/><circle cx="7" cy="17" r="1.5"/><circle cx="17" cy="17" r="1.5"/></svg>"""
    svg_p = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="4"/><path d="M9 17V7h4a3 3 0 0 1 0 6H9"/></svg>"""

    for r in results:
        is_occ = (r["prediction"] == 1)
        bay_class = "bay-occupied" if is_occ else "bay-free"
        s_id = r["id"]
        conf = int(r.get("confidence", 1.0) * 100)
        icon_svg = svg_car if is_occ else svg_p

        bays_html.append(f"""
        <div class="parking-bay {bay_class}" title="Space #{s_id}: {'OCCUPIED' if is_occ else 'FREE'} ({conf}% certainty)">
          <span class="bay-id">#{s_id}</span>
          <span class="bay-icon">{icon_svg}</span>
          <span class="bay-conf">{conf}%</span>
        </div>
        """)

    grid_content = "".join(bays_html)
    st.markdown(f"""
    <div class="parking-bay-grid">
      {grid_content}
    </div>
    <div class="parking-bay-legend">
      <span class="legend-chip"><span class="legend-box green"></span> Available Free Space (Pulsing)</span>
      <span class="legend-chip"><span class="legend-box red"></span> Occupied Space</span>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_filmstrip() -> None:
    """Render horizontal step-by-step pipeline filmstrip cards with arrow connectors."""
    steps = [
        ("1", "Perspective Homography", "Warps 4-point polygon to 64x64 canonical patch"),
        ("2", "Grayscale Conversion", "Eliminates chromatic noise, isolates luminance"),
        ("3", "Gaussian Noise Filter", "Suppresses asphalt texture noise with 5x5 kernel"),
        ("4", "CLAHE Equalization", "Adaptive contrast enhancement prevents shadow blowout"),
        ("5", "Canny Edge Density", "Captures sharp car contours, grilles and seams"),
        ("6", "Background Subtraction", "Calculates absolute pixel difference vs. empty model"),
        ("7", "Decision Logic", "Weighted linear scoring evaluates threshold")
    ]

    cards_html = []
    for i, (num, title, desc) in enumerate(steps):
        cards_html.append(f"""
        <div class="filmstrip-card">
          <div class="filmstrip-badge">STAGE 0{num}</div>
          <div class="filmstrip-title">{title}</div>
          <div class="filmstrip-caption">{desc}</div>
        </div>
        """)
        if i < len(steps) - 1:
            cards_html.append("""<div class="filmstrip-arrow">&rarr;</div>""")

    strip_content = "".join(cards_html)
    st.markdown(f"""
    <div class="filmstrip-container">
      {strip_content}
    </div>
    """, unsafe_allow_html=True)


def render_next_tab_card(next_name: str, hint: str) -> None:
    """Render slim next step card at the bottom of a tab."""
    st.markdown(f"""
    <div class="next-action-card">
      <div class="next-action-meta">
        <span class="next-action-tag">Next Section</span>
        <span class="next-action-title">Proceed to <b>{next_name}</b></span>
      </div>
      <div class="next-action-hint">{hint} &rarr;</div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(title: str, description: str, hint: str = "") -> None:
    """Render friendly glassmorphism empty state placeholder (no emojis)."""
    hint_html = f'<div class="empty-state-hint">Hint: {hint}</div>' if hint else ""
    svg_icon = """<svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""

    st.markdown(f"""
    <div class="empty-state-box">
      <div class="empty-state-icon">{svg_icon}</div>
      <div class="empty-state-title">{title}</div>
      <div class="empty-state-desc">{description}</div>
      {hint_html}
    </div>
    """, unsafe_allow_html=True)


def render_footer() -> None:
    """Render minimal centered product footer on every page."""
    st.markdown("""
    <div class="app-footer">
      <span>ParkLens | Fundamentals of Computer Vision mini project | Dataset: PKLot (Almeida et al., 2015)</span>
    </div>
    """, unsafe_allow_html=True)
