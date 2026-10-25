"""
ParkLens – Classical Computer Vision Parking Occupancy Detector
Presentation & Demo Web Application built with Streamlit.

Run with:
    streamlit run app.py
"""

import sys
import os
import io
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR, PKLOT_DIR, INDEX_FILE, OUTPUTS_DIR,
    PARAMS_FILE, RESULTS_FILE, METRICS_FILE, LOTS, WEATHERS
)
from src.acquisition import parse_pklot_xml, load_dataset_index
from src.preprocessing import preprocess_patch, to_grayscale, apply_blur, apply_clahe
from src.features import extract_features, edge_density, background_diff_score
from src.classifier import classify_spaces_in_image, load_tuned_params, rule_based_classify
from src.evaluation import compute_metrics
from src.visualization import draw_parking_overlay, make_pipeline_figure
from src.utils import warp_polygon_to_patch
from ui.components import (
    inject_custom_styles,
    render_hero_header,
    render_context_bar,
    render_status_strip,
    render_page_header,
    render_how_to_use_banner,
    render_kpi_card,
    render_space_legend,
    render_parking_bays,
    render_pipeline_filmstrip,
    render_next_tab_card,
    render_empty_state,
    render_footer
)
from ui.charts import create_occupancy_donut, create_feature_bars, create_timeline_chart


# ─── Streamlit Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="ParkLens - Classical CV Parking Detector",
    page_icon=str(PROJECT_ROOT / "assets" / "favicon.png") if (PROJECT_ROOT / "assets" / "favicon.png").exists() else "P",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Custom CSS & Dynamic Background Textures ──────────────────────────
inject_custom_styles()


# ─── Cached Resource Loaders ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_cached_index() -> List[Dict[str, Any]]:
    """Load normalized dataset index."""
    if INDEX_FILE.exists():
        return load_dataset_index(INDEX_FILE)
    return []


@st.cache_resource(show_spinner=False)
def get_cached_ref_patches() -> Dict[str, np.ndarray]:
    """Load per-space empty reference patches."""
    ref_path = DATA_DIR / "ref_patches.pkl"
    if ref_path.exists():
        with open(ref_path, "rb") as f:
            return pickle.load(f)
    return {}


@st.cache_data(show_spinner=False)
def get_cached_results() -> Optional[Dict[str, Any]]:
    """Load evaluation test benchmark results."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_image_bgr(path_str: str) -> Optional[np.ndarray]:
    """Load image from relative or absolute path string."""
    full_path = PROJECT_ROOT / path_str
    if not full_path.exists():
        return None
    return cv2.imread(str(full_path))


# ─── Global State & Metadata ────────────────────────────────────────────────
dataset_index = get_cached_index()
ref_patches = get_cached_ref_patches()
eval_results = get_cached_results()
is_demo_data = len(dataset_index) == 0 or dataset_index[0].get("is_synthetic", False)


# ─── Sidebar Controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-header">QUICK DEMO PRESETS</div>', unsafe_allow_html=True)
    col_d1, col_d2, col_d3 = st.columns(3)
    preset_choice = None
    with col_d1:
        if st.button("Sunny", use_container_width=True, help="Load Sunny day capture"):
            preset_choice = ("PUCPR", "Sunny")
    with col_d2:
        if st.button("Cloudy", use_container_width=True, help="Load Cloudy day capture"):
            preset_choice = ("PUCPR", "Cloudy")
    with col_d3:
        if st.button("Rainy", use_container_width=True, help="Load Rainy day capture"):
            preset_choice = ("PUCPR", "Rainy")

    with st.expander("1. DATA INGESTION", expanded=True):
        source_mode = st.radio(
            "Source Mode",
            ["PKLot Dataset", "Upload Custom Image"],
            label_visibility="collapsed",
            help="Select a benchmark image from the test set or upload your own file"
        )

        selected_image_record = None
        uploaded_image_bgr = None
        uploaded_spaces = []

        if source_mode == "PKLot Dataset":
            if not dataset_index:
                st.warning("No dataset index found. Run python main.py prepare or download data.")
            else:
                available_lots = sorted(list(set(r["lot"] for r in dataset_index)))
                lot_default_idx = 0
                if preset_choice and preset_choice[0] in available_lots:
                    lot_default_idx = available_lots.index(preset_choice[0])

                selected_lot = st.selectbox("Parking Lot", available_lots, index=lot_default_idx)

                available_weathers = sorted(list(set(
                    r["weather"] for r in dataset_index if r["lot"] == selected_lot
                )))
                weather_default_idx = 0
                if preset_choice and preset_choice[1] in available_weathers:
                    weather_default_idx = available_weathers.index(preset_choice[1])

                selected_weather = st.selectbox("Weather Condition", available_weathers, index=weather_default_idx)

                matching_images = [
                    r for r in dataset_index
                    if r["lot"] == selected_lot and r["weather"] == selected_weather
                ]

                img_options = [r["datetime_str"] for r in matching_images]
                selected_img_name = st.selectbox("Timestamp", img_options)
                selected_image_record = next(
                    (r for r in matching_images if r["datetime_str"] == selected_img_name),
                    matching_images[0] if matching_images else None
                )
        else:
            uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            uploaded_xml = st.file_uploader("Optional: Upload XML Contour File", type=["xml"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                uploaded_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if uploaded_xml is not None:
                    temp_xml = PROJECT_ROOT / "outputs" / "temp_upload.xml"
                    temp_xml.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_xml, "wb") as f:
                        f.write(uploaded_xml.read())
                    uploaded_spaces = parse_pklot_xml(temp_xml)
                else:
                    st.info("No XML supplied. Full-lot visual overlays require polygon coordinates.")

    with st.expander("2. DETECTION METHOD", expanded=True):
        st.caption("Classical CV: Homography Rectification + Gaussian Blur + CLAHE + Canny Edges + Background Subtraction.")
        method_choice = st.selectbox(
            "Classifier Architecture",
            ["Rule-Based Weighted Linear (Main)", "Classical Logistic Regression"],
            help="ParkLens uses an explainable rule-based combination of edge density and background subtraction"
        )

    current_lot = selected_image_record["lot"] if selected_image_record else "PUCPR"
    current_weather = selected_image_record["weather"] if selected_image_record else "Sunny"
    tuned_params = load_tuned_params(PARAMS_FILE, lot=current_lot, weather=current_weather)

    with st.expander("3. MODEL TUNING", expanded=False):
        if st.button("Reset to Tuned Values", use_container_width=True, help="Load empirically tuned values for this lot and weather"):
            st.session_state["canny_low"] = tuned_params.get("canny_low", 40)
            st.session_state["canny_high"] = tuned_params.get("canny_high", 120)
            st.session_state["diff_threshold"] = tuned_params.get("diff_threshold", 25)
            st.session_state["decision_threshold"] = float(tuned_params.get("decision_threshold", 0.25))
            st.session_state["blur_k"] = 5
            st.session_state["clahe_clip"] = 2.0

        canny_low = st.slider(
            "Canny Low Hysteresis", 10, 150,
            st.session_state.get("canny_low", tuned_params.get("canny_low", 40)),
            step=5, help="Lower gradient threshold for edge continuity"
        )
        canny_high = st.slider(
            "Canny High Hysteresis", 50, 250,
            st.session_state.get("canny_high", tuned_params.get("canny_high", 120)),
            step=5, help="Upper gradient threshold for strong edge seed detection"
        )
        diff_threshold = st.slider(
            "Background Diff Threshold", 10, 80,
            st.session_state.get("diff_threshold", tuned_params.get("diff_threshold", 25)),
            step=2, help="Absolute intensity threshold against canonical empty reference"
        )
        decision_threshold = st.slider(
            "Decision Threshold", 0.10, 0.60,
            float(st.session_state.get("decision_threshold", tuned_params.get("decision_threshold", 0.25))),
            step=0.02, help="Cutoff score: values above are declared OCCUPIED"
        )

        blur_k = st.slider("Gaussian Blur Kernel", 1, 11, st.session_state.get("blur_k", 5), step=2)
        clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, st.session_state.get("clahe_clip", 2.0), step=0.5)

    current_params = {
        "canny_low": canny_low,
        "canny_high": canny_high,
        "diff_threshold": diff_threshold,
        "decision_threshold": decision_threshold,
        "blur_kernel": blur_k,
        "clahe_clip": clahe_clip,
        "edge_weight": 0.5,
        "diff_weight": 0.5
    }

    with st.expander("4. DISPLAY & DIAGNOSTICS", expanded=False):
        show_ids = st.checkbox("Show Space IDs", value=False, help="Render numeric ID at space centroid")
        show_confidence = st.checkbox("Show Confidence %", value=False, help="Display certainty rating per space")
        highlight_errors = st.checkbox("Highlight Errors (Yellow)", value=True, help="Outline False Positives and Negatives in yellow")

    st.markdown("---")
    pres_mode = st.toggle("Presentation Mode (Projector)", value=st.session_state.get("presentation_mode", False), help="Enlarges typography and metrics for external displays")
    if pres_mode != st.session_state.get("presentation_mode", False):
        st.session_state["presentation_mode"] = pres_mode
        st.rerun()

    if st.button("Reset All Settings", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ─── Load Active Image & Run Pipeline ───────────────────────────────────────
active_img = None
active_spaces = []
active_lot = "PUCPR"
active_filename = ""

if source_mode == "PKLot Dataset" and selected_image_record:
    active_img = load_image_bgr(selected_image_record["image_path"])
    active_spaces = selected_image_record.get("spaces", [])
    active_lot = selected_image_record.get("lot", "PUCPR")
    active_filename = Path(selected_image_record["image_path"]).name
elif source_mode == "Upload Custom Image" and uploaded_image_bgr is not None:
    active_img = uploaded_image_bgr
    active_spaces = uploaded_spaces
    active_lot = "UPLOADED"
    active_filename = "custom_upload.jpg"


detection_results = []
if active_img is not None and active_spaces:
    detection_results = classify_spaces_in_image(
        active_img, active_spaces, active_lot, ref_patches, current_params
    )


# ─── Render Hero Header & Top Navigation Bars ───────────────────────────────
render_hero_header(is_synthetic=is_demo_data)

render_context_bar(
    section="Live Detection",
    lot=active_lot,
    weather=current_weather,
    filename=active_filename if active_filename else "No image loaded"
)

render_status_strip(
    is_synthetic=is_demo_data,
    space_count=len(active_spaces)
)


# ─── Tab Navigation (No Emojis, Clean Sticky Tabs) ──────────────────────────
tab_live, tab_pipeline, tab_map, tab_timeline, tab_perf, tab_about = st.tabs([
    "Live Detection",
    "Pipeline Explorer",
    "Parking Map",
    "Timeline",
    "Performance Dashboard",
    "About"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE DETECTION
# ═══════════════════════════════════════════════════════════════════════════
with tab_live:
    render_page_header(
        title="Live Detection",
        description="Pick a lot, weather and time in the sidebar to see which spaces are free, or upload custom imagery."
    )

    render_how_to_use_banner()

    if active_img is None:
        render_empty_state(
            title="No Parking Lot Image Loaded",
            description="Select a parking lot from the sidebar or click one of the Quick Demo presets above.",
            hint="Try clicking 'Sunny' in the sidebar for an instant demonstration."
        )
    elif not active_spaces:
        st.warning("Parking lot image loaded, but no slot polygon coordinates were detected.")
        st.image(cv2.cvtColor(active_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        col_img, col_metrics = st.columns([1.8, 1.0], gap="large")

        with col_img:
            # Inspection toggle (Raw vs Annotated)
            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                view_mode = st.radio(
                    "Display View",
                    ["Annotated Overlay", "Raw Camera Capture"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

            annotated_bgr = draw_parking_overlay(
                active_img, detection_results,
                show_ids=show_ids,
                show_confidence=show_confidence,
                highlight_errors=highlight_errors
            )

            display_target = annotated_bgr if view_mode == "Annotated Overlay" else active_img

            # Framed Glass Card
            st.markdown(f"""
            <div class="framed-image-box">
              <div class="image-meta-bar">
                <span><b>Lot:</b> {active_lot} &bull; <b>Weather:</b> {current_weather}</span>
                <span class="image-meta-pill">{active_filename} ({active_img.shape[1]}x{active_img.shape[0]})</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                cv2.cvtColor(display_target, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            render_space_legend(highlight_errors=highlight_errors)

            # Export actions
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                _, png_buf = cv2.imencode(".png", annotated_bgr)
                st.download_button(
                    "Export Annotated Frame (PNG)",
                    data=png_buf.tobytes(),
                    file_name=f"parklens_{active_lot}_{active_filename}.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_d2:
                rows = []
                for r in detection_results:
                    rows.append({
                        "Space_ID": r["id"],
                        "Prediction": "Occupied" if r["prediction"] == 1 else "Free",
                        "Confidence": f"{r['confidence']*100:.1f}%",
                        "Score": f"{r['combined_score']:.4f}",
                        "Ground_Truth": "Occupied" if r.get("ground_truth") == 1 else "Free" if r.get("ground_truth") == 0 else "N/A",
                        "Correct": r.get("is_correct"),
                        "Edge_Density": f"{r['features']['edge_density']:.4f}",
                        "Diff_Score": f"{r['features']['diff_score']:.4f}"
                    })
                df_results = pd.DataFrame(rows)
                st.download_button(
                    "Export Space Data (CSV)",
                    data=df_results.to_csv(index=False).encode("utf-8"),
                    file_name=f"parklens_spaces_{active_filename}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Collapsible Detailed Space Breakdown Table
            with st.expander("Detailed Space Breakdown Table", expanded=False):
                st.dataframe(df_results, use_container_width=True, hide_index=True)

        with col_metrics:
            total_s = len(detection_results)
            occ_s = sum(1 for r in detection_results if r["prediction"] == 1)
            free_s = total_s - occ_s
            occ_pct = (occ_s / total_s * 100) if total_s > 0 else 0

            # 4 High-contrast KPI Cards
            st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
            kpi_c1, kpi_c2 = st.columns(2)
            with kpi_c1:
                render_kpi_card("Total Spaces", f"{total_s}", "teal", "Monitored capacity")
                st.write("")
                render_kpi_card("Available", f"{free_s}", "green", "Free for parking")
            with kpi_c2:
                render_kpi_card("Occupied", f"{occ_s}", "red", "Vehicles detected")
                st.write("")
                render_kpi_card("Occupancy", f"{occ_pct:.0f}%", "amber", "Capacity utilization")
            st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            # Interactive Donut Chart with center total
            st.plotly_chart(
                create_occupancy_donut(free_s, occ_s),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            # Frame Diagnostics
            has_gt = any(r.get("ground_truth") is not None for r in detection_results)
            if has_gt:
                y_t = [r["ground_truth"] for r in detection_results if r.get("ground_truth") is not None]
                y_p = [r["prediction"] for r in detection_results if r.get("ground_truth") is not None]
                frame_m = compute_metrics(y_t, y_p)

                has_occ_gt = (frame_m["tp"] + frame_m["fn"]) > 0
                recall_display = f"{frame_m['recall']*100:.1f}%" if has_occ_gt else "N/A"
                recall_tooltip = "Cars identified" if has_occ_gt else "Undefined: no occupied spaces in ground truth for this frame"

                has_occ_pred = (frame_m["tp"] + frame_m["fp"]) > 0
                prec_display = f"{frame_m['precision']*100:.1f}%" if has_occ_pred else "N/A"
                prec_tooltip = "Occupied alerts accuracy" if has_occ_pred else "Undefined: no spaces predicted occupied in this frame"

                st.markdown("""
                <div class="glass-panel" style="margin-top:10px;padding:12px 16px;">
                  <div style="font-size:0.75rem;font-weight:750;letter-spacing:0.08em;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">
                    Single-Frame Ground Truth Diagnostics
                  </div>
                </div>
                """, unsafe_allow_html=True)

                f_c1, f_c2, f_c3 = st.columns(3)
                with f_c1:
                    render_kpi_card("Accuracy", f"{frame_m['accuracy']*100:.1f}%", "teal", "Exact match")
                with f_c2:
                    render_kpi_card("Recall", recall_display, "green", "Cars identified", tooltip=recall_tooltip)
                with f_c3:
                    render_kpi_card("Precision", prec_display, "amber", "Alerts trust", tooltip=prec_tooltip)

    render_next_tab_card(
        next_name="Pipeline Explorer",
        hint="Inspect the 6-stage classical image processing transformation sequence"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: PIPELINE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    render_page_header(
        title="Pipeline Explorer",
        description="Inspect how classical image processing isolates vehicle contours and calculates decision scores without neural networks."
    )

    render_pipeline_filmstrip()

    if active_img is not None and active_spaces:
        space_ids = [str(s["id"]) for s in active_spaces]
        selected_space_id = st.selectbox("Inspect Individual Space ID:", space_ids, index=0)

        target_sp = next((s for s in active_spaces if str(s["id"]) == selected_space_id), active_spaces[0])
        patch_bgr = warp_polygon_to_patch(active_img, target_sp["polygon"])
        ref_patch = ref_patches.get(f"{active_lot}_{selected_space_id}")

        feat_data = extract_features(
            patch_bgr=patch_bgr,
            ref_patch_bgr=ref_patch,
            canny_low=current_params["canny_low"],
            canny_high=current_params["canny_high"],
            diff_threshold=current_params["diff_threshold"],
            blur_k=current_params["blur_kernel"],
            clahe_clip=current_params["clahe_clip"]
        )

        pred, conf, score = rule_based_classify(
            feat_data,
            edge_weight=current_params["edge_weight"],
            diff_weight=current_params["diff_weight"],
            decision_threshold=current_params["decision_threshold"]
        )

        gt = target_sp.get("occupied")

        comp_img = make_pipeline_figure(patch_bgr, ref_patch, current_params)
        st.markdown(f"""
        <div class="framed-image-box">
          <div class="image-meta-bar">
            <span><b>Pipeline Visual Transformation Sequence</b> – Space #{selected_space_id}</span>
            <span class="image-meta-pill">Score: {score:.3f} | Threshold: {current_params['decision_threshold']:.2f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        col_insp1, col_insp2, col_insp3 = st.columns([1, 1.2, 1.2], gap="medium")

        with col_insp1:
            st.markdown("**Observation vs. Reference**")
            st.image(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB), caption="Warped Observation (64x64)", width=150)
            if ref_patch is not None:
                st.image(cv2.cvtColor(ref_patch, cv2.COLOR_BGR2RGB), caption="Empty Baseline Model", width=150)
            else:
                st.caption("No canonical reference patch available")

        with col_insp2:
            st.markdown("**Extracted Classical Features**")
            st.plotly_chart(create_feature_bars(feat_data), use_container_width=True, config={"displayModeBar": False})

            st.markdown(f"""
            - **Edge Density:** `{feat_data['edge_density']:.4f}`
            - **Diff Score:** `{feat_data['diff_score']:.4f}`
            - **Laplacian Variance:** `{feat_data['laplacian_variance']:.1f}`
            - **Intensity Std Dev:** `{feat_data['gray_std']:.2f}`
            """)

        with col_insp3:
            st.markdown("**Classifier Decision Logic**")
            status_text = "OCCUPIED" if pred == 1 else "FREE"
            status_color = "red" if pred == 1 else "green"

            render_kpi_card("Classifier Output", status_text, status_color, f"Certainty: {conf*100:.1f}%")
            st.write("")
            render_kpi_card("Combined Score", f"{score:.3f}", "teal", f"Cutoff: {current_params['decision_threshold']:.2f}")

            if gt is not None:
                gt_text = "OCCUPIED" if gt == 1 else "FREE"
                is_correct = (pred == gt)
                verdict_color = "#22c55e" if is_correct else "#ef4444"
                verdict_label = "Correct Classification" if is_correct else "Classification Error"
                st.markdown(f"""
                <div class="glass-panel" style="margin-top:10px;padding:10px 14px;border-left:4px solid {verdict_color};">
                  <b>Ground Truth:</b> <code>{gt_text}</code><br>
                  <b>Evaluation:</b> {verdict_label}
                </div>
                """, unsafe_allow_html=True)
    else:
        render_empty_state("Pipeline Explorer Inactive", "Select an image from the sidebar to inspect individual spaces.")

    render_next_tab_card(
        next_name="Parking Map",
        hint="View top-down parking space allocation and nearest stall wayfinding"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: PARKING MAP
# ═══════════════════════════════════════════════════════════════════════════
with tab_map:
    render_page_header(
        title="Parking Map",
        description="Top-down architectural parking space matrix with free space indicators and nearest available routing."
    )

    if detection_results:
        col_grid, col_rec = st.columns([2, 1], gap="large")

        with col_grid:
            st.markdown("##### Architectural Parking Matrix")
            render_parking_bays(detection_results)

        with col_rec:
            st.markdown("##### Nearest Free Spaces")
            free_spots = [r for r in detection_results if r["prediction"] == 0]

            if not free_spots:
                st.error("Facility Full: No vacant parking spaces available.")
            else:
                def get_cy(r):
                    pts = np.array(r["polygon"])
                    return pts[:, 1].mean()

                sorted_free = sorted(free_spots, key=get_cy)

                st.success(f"{len(free_spots)} spaces currently available")
                for i, s in enumerate(sorted_free[:8]):
                    conf_pct = int(s.get("confidence", 1.0) * 100)
                    st.markdown(
                        f"""
                        <div class="glass-panel" style="padding:10px 14px;margin-bottom:8px;border-left:4px solid #22c55e;display:flex;justify-content:space-between;align-items:center;">
                          <span><b>#{i+1}. Space #{s['id']}</b></span>
                          <span style="color:#4ade80;font-size:0.8rem;font-weight:600;">AVAILABLE ({conf_pct}%)</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        render_empty_state("No Map Data Available", "Load an image in the sidebar to populate the parking bay matrix.")

    render_next_tab_card(
        next_name="Timeline",
        hint="Track temporal occupancy changes across diurnal sequences"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: TIMELINE (TIME-LAPSE)
# ═══════════════════════════════════════════════════════════════════════════
with tab_timeline:
    render_page_header(
        title="Timeline",
        description="Track diurnal temporal sequence and parking lot occupancy transitions across chronological captures."
    )

    if dataset_index:
        cohorts = defaultdict(list)
        for r in dataset_index:
            key = f"{r['lot']} – {r.get('date', 'Unknown')}"
            cohorts[key].append(r)

        valid_cohorts = {k: v for k, v in cohorts.items() if len(v) >= 5}
        if not valid_cohorts:
            valid_cohorts = cohorts

        selected_cohort = st.selectbox("Select Day & Lot Sequence:", list(valid_cohorts.keys()))
        sequence_frames = sorted(valid_cohorts[selected_cohort], key=lambda x: x.get("datetime_str", ""))[:30]

        timeline_data = []
        progress_bar = st.progress(0, text="Analyzing temporal captures...")

        for idx, frame_rec in enumerate(sequence_frames):
            img_bgr = load_image_bgr(frame_rec["image_path"])
            if img_bgr is not None and frame_rec.get("spaces"):
                res = classify_spaces_in_image(
                    img_bgr, frame_rec["spaces"], frame_rec["lot"], ref_patches, current_params
                )
                occ = sum(1 for r in res if r["prediction"] == 1)
                free = len(res) - occ
                time_str = frame_rec["datetime_str"].split("_")[-3:] if "_" in frame_rec["datetime_str"] else [frame_rec["datetime_str"]]
                time_lbl = ":".join(time_str[-3:]) if len(time_str) >= 3 else frame_rec["datetime_str"]
                timeline_data.append({
                    "time_label": time_lbl,
                    "occupied": occ,
                    "free": free,
                    "total": len(res),
                    "image_path": frame_rec["image_path"],
                    "spaces": frame_rec["spaces"],
                    "results": res
                })
            progress_bar.progress((idx + 1) / len(sequence_frames))

        progress_bar.empty()

        if timeline_data:
            st.markdown('<div class="glass-panel" style="padding:14px;margin-bottom:1rem;">', unsafe_allow_html=True)
            st.plotly_chart(create_timeline_chart(timeline_data), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            frame_idx = st.slider("Scrub Temporal Frame Sequence:", 0, len(timeline_data) - 1, 0)
            cur_frame = timeline_data[frame_idx]

            scrub_img = load_image_bgr(cur_frame["image_path"])
            if scrub_img is not None:
                annotated_scrub = draw_parking_overlay(scrub_img, cur_frame["results"])
                st.markdown(f"""
                <div class="framed-image-box">
                  <div class="image-meta-bar">
                    <span><b>Timestamp:</b> {cur_frame['time_label']}</span>
                    <span class="image-meta-pill">Available: {cur_frame['free']} | Occupied: {cur_frame['occupied']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(
                    cv2.cvtColor(annotated_scrub, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
    else:
        render_empty_state("Timeline Inactive", "Dataset index needed to run temporal time-lapse analysis.")

    render_next_tab_card(
        next_name="Performance Dashboard",
        hint="Review test-split benchmark metrics and quantitative validation results"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: PERFORMANCE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tab_perf:
    render_page_header(
        title="Performance Dashboard",
        description="Benchmark results strictly evaluated on the unseen 20% test split from the PKLot dataset."
    )

    if eval_results is not None:
        o = eval_results.get("overall", {})
        m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
        with m_c1:
            render_kpi_card("Accuracy", f"{o.get('accuracy', 0)*100:.1f}%", "teal", "Overall accuracy")
        with m_c2:
            render_kpi_card("F1-Score", f"{o.get('f1', 0)*100:.1f}%", "green", "Harmonic mean")
        with m_c3:
            render_kpi_card("Recall", f"{o.get('recall', 0)*100:.1f}%", "teal", "Cars detected")
        with m_c4:
            render_kpi_card("Specificity", f"{o.get('specificity', 0)*100:.1f}%", "teal", "Empty recognized")
        with m_c5:
            render_kpi_card("Precision", f"{o.get('precision', 0)*100:.1f}%", "teal", "Declaration trust")
        with m_c6:
            render_kpi_card("ROC AUC", f"{o.get('auc', 0):.3f}", "amber", "Separation index")

        st.markdown("---")
        col_heat, col_roc, col_bar = st.columns(3, gap="medium")

        with col_heat:
            st.markdown("##### Confusion Matrix")
            cm_img_path = OUTPUTS_DIR / "confusion_matrix.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), use_container_width=True)

        with col_roc:
            st.markdown("##### ROC Curves (AUC)")
            roc_img_path = OUTPUTS_DIR / "roc_curves.png"
            if roc_img_path.exists():
                st.image(str(roc_img_path), use_container_width=True)

        with col_bar:
            st.markdown("##### Weather Accuracy")
            bar_img_path = OUTPUTS_DIR / "weather_accuracy.png"
            if bar_img_path.exists():
                st.image(str(bar_img_path), use_container_width=True)

        st.markdown("---")
        st.markdown("##### Breakdown Table per Lot & Weather Condition")
        csv_path = OUTPUTS_DIR / "results.csv"
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path)
            st.dataframe(df_csv, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### Error Analysis & Failure Cases Gallery")
        failures = eval_results.get("sample_failures", [])
        if failures:
            f_cols = st.columns(3)
            for idx, fail in enumerate(failures[:6]):
                col_i = f_cols[idx % 3]
                with col_i:
                    err_type = fail["error_type"]
                    badge_c = "#ef4444" if err_type == "False Positive" else "#f59e0b"
                    st.markdown(f"""
                    <div class="failure-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <span style="font-weight:700;color:{badge_c};font-size:0.85rem;">{err_type}</span>
                        <span class="image-meta-pill">Score: {fail['combined_score']}</span>
                      </div>
                      <div style="font-size:0.78rem;color:#CBD5E1;margin-bottom:8px;">
                        <b>Space #{fail['space_id']}</b> &bull; {fail['lot']} ({fail['weather']})
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    f_img = load_image_bgr(fail["image_path"])
                    if f_img is not None:
                        idx_match = next((r for r in dataset_index if r["image_path"] == fail["image_path"]), None)
                        if idx_match:
                            sp_match = next((s for s in idx_match["spaces"] if str(s["id"]) == str(fail["space_id"])), None)
                            if sp_match:
                                p_crop = warp_polygon_to_patch(f_img, sp_match["polygon"])
                                st.image(cv2.cvtColor(p_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption("Low-angle solar shadow or pavement luminance artifact.")
    else:
        render_empty_state("Benchmark Outputs Not Found", "Execute 'python main.py evaluate' to generate test set metrics.")

    render_next_tab_card(
        next_name="About",
        hint="Learn about the system architecture and academic references"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: ABOUT
# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    render_page_header(
        title="About ParkLens",
        description="Transparent classical computer vision architecture, methodology, and citations."
    )

    st.markdown("""
    <div class="about-backdrop-card">
      <div class="about-overlay"></div>
      <div class="about-content">
        <h3 style="color:#FFFFFF;margin-top:0;">About ParkLens</h3>
        <p style="color:#CBD5E1;max-width:850px;line-height:1.6;font-size:0.9rem;">
          <b>ParkLens</b> is an explainable, classical computer vision system engineered for the university course
          <i>Fundamentals of Computer Vision</i>. Unlike black-box deep neural networks (such as YOLO or Mask R-CNN),
          ParkLens relies entirely on transparent, mathematically verifiable image processing primitives.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_ab1, col_ab2 = st.columns([1.2, 1], gap="large")

    with col_ab1:
        st.markdown("""
        <div class="glass-panel">
          <h4 style="color:#22D3EE;margin-top:0;">Core Methodological Architecture</h4>
          <ol style="color:#CBD5E1;line-height:1.7;padding-left:20px;font-size:0.86rem;">
            <li><b>Perspective Homography Rectification:</b> Warps quadrilateral slot polygons to normalized $64 \\times 64$ patches using 4-point projective transformation.</li>
            <li><b>Illumination Compensation (CLAHE):</b> Contrast Limited Adaptive Histogram Equalization suppresses harsh shadows and sun glints in localized $8 \\times 8$ tiles.</li>
            <li><b>Canny Gradient Edge Analysis:</b> Exploits high-frequency vehicle contours (hood seams, grilles, license plates) vs. smooth pavement.</li>
            <li><b>Background Subtraction:</b> Computes pixel discrepancy against a per-space temporal median empty reference model.</li>
            <li><b>Stratified Parameter Optimization:</b> Grid-searches optimal thresholds per parking lot and weather condition on the training split.</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_ab2:
        st.markdown("""
        <div class="glass-panel">
          <h4 style="color:#22D3EE;margin-top:0;">Academic Citations & Credits</h4>
          <p style="color:#CBD5E1;font-size:0.82rem;line-height:1.6;">
            <b>1. PKLot Benchmark Dataset:</b><br>
            Almeida, P. R., Oliveira, L. S., Britto, A. S., Silva, E. J., & Koerich, A. L. (2015).
            <i>PKLot – A robust dataset for parking lot classification.</i> Expert Systems with Applications, 42(11), 4937-4949.
            (Licensed under CC BY 4.0).
          </p>
          <p style="color:#CBD5E1;font-size:0.82rem;line-height:1.6;">
            <b>2. OpenCV (Open Source Computer Vision):</b><br>
            Bradski, G. (2000). <i>The OpenCV Library.</i> Dr. Dobb's Journal of Software Tools.
          </p>
          <p style="color:#CBD5E1;font-size:0.82rem;line-height:1.6;">
            <b>3. VisionParkDetect:</b><br>
            Inspirational reference for classical edge and background subtraction principles.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("##### System Computational Flowchart")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR;
      bgcolor="transparent";
      node [shape=box, style="filled,rounded", color="#334155", fillcolor="#0F172A", fontname="sans-serif", fontcolor="#F8FAFC"];
      edge [color="#22D3EE", fontname="sans-serif", fontcolor="#94A3B8"];
      
      Input [label="Input Lot Image\\n+ Polygons", fillcolor="#0369A1", fontcolor="#FFFFFF"];
      Warp [label="Perspective Warp\\n(64x64 Patch)"];
      CLAHE [label="Gaussian Blur\\n+ CLAHE"];
      Edges [label="Canny Edge\\nDetection"];
      Diff [label="Background\\nSubtraction"];
      Decision [label="Weighted Scoring\\n+ Threshold", fillcolor="#0F766E", fontcolor="#FFFFFF"];
      Output [label="Free / Occupied\\nAnnotation", fillcolor="#15803D", fontcolor="#FFFFFF"];
      
      Input -> Warp -> CLAHE;
      CLAHE -> Edges -> Decision;
      CLAHE -> Diff -> Decision;
      Decision -> Output;
    }
    """)


# ─── Centered Universal Footer ──────────────────────────────────────────────
render_footer()
