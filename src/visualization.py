"""
src/visualization.py – Rich visual representations for CLI and UI.

Functions:
  1. draw_parking_overlay: Overlay green/red polygons and banner on lot image.
  2. make_pipeline_figure: Horizontal composite of intermediate pipeline steps.
  3. plot_confusion_matrix_heatmap: Matplotlib annotated 2x2 confusion matrix.
  4. plot_roc_curves: Matplotlib ROC curves with per-lot AUC badges.
  5. plot_weather_accuracy_bars: Grouped bar chart comparing performance across weather.
  6. save_all_evaluation_figures: Generate and save all artifact plots to outputs/.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FREE_COLOR, OCCUPIED_COLOR, ERROR_COLOR,
    OUTPUTS_DIR, FONT_SCALE, LINE_THICKNESS
)


def draw_parking_overlay(
    image_bgr: np.ndarray,
    results: List[Dict[str, Any]],
    show_ids: bool = False,
    show_confidence: bool = False,
    highlight_errors: bool = False,
    alpha: float = 0.35
) -> np.ndarray:
    """
    Render colored polygon overlays and top dashboard banner onto full parking lot image.
    
    Args:
        image_bgr: Original full lot image.
        results: List of space result dictionaries (from classify_spaces_in_image).
        show_ids: If True, draws integer slot identifier in space center.
        show_confidence: If True, draws confidence percentage.
        highlight_errors: If True and ground_truth exists, outlines FP/FN in yellow.
        alpha: Transparency factor for filled polygon highlights.
        
    Returns:
        Annotated BGR image with translucent overlays and top banner.
    """
    overlay = image_bgr.copy()
    output = image_bgr.copy()

    total_spaces = len(results)
    occupied_count = sum(1 for r in results if r["prediction"] == 1)
    free_count = total_spaces - occupied_count

    for r in results:
        poly = np.array(r["polygon"], dtype=np.int32)
        pred = r["prediction"]
        conf = r.get("confidence", 1.0)
        gt = r.get("ground_truth")
        is_error = (highlight_errors and gt is not None and pred != gt)

        # Base color: Green for Free (0), Red for Occupied (1)
        base_color = OCCUPIED_COLOR if pred == 1 else FREE_COLOR
        border_color = ERROR_COLOR if is_error else base_color
        thickness = LINE_THICKNESS + 2 if is_error else LINE_THICKNESS

        # Translucent fill
        cv2.fillPoly(overlay, [poly], base_color)
        # Border
        cv2.polylines(output, [poly], isClosed=True, color=border_color, thickness=thickness)

        # Labels (Space ID or Confidence)
        if show_ids or show_confidence:
            # Centroid of polygon
            M = cv2.moments(poly)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())

            text_parts = []
            if show_ids:
                text_parts.append(str(r["id"]))
            if show_confidence:
                text_parts.append(f"{int(conf * 100)}%")
            label = " ".join(text_parts)

            # Small text background box for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(output, (cx - tw // 2 - 2, cy - th // 2 - 2), (cx + tw // 2 + 2, cy + th // 2 + 2), (0, 0, 0), -1)
            cv2.putText(output, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend translucent fill with image
    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)

    # Draw Top Dashboard Header Banner
    h, w = output.shape[:2]
    banner_h = 44
    banner_bg = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner_bg[:] = (20, 24, 30)  # Dark slate gray
    cv2.addWeighted(banner_bg, 0.85, output[:banner_h], 0.15, 0, output[:banner_h])

    # Banner Text
    occ_pct = (occupied_count / total_spaces * 100) if total_spaces > 0 else 0
    text_free = f"FREE: {free_count}"
    text_occ = f"OCCUPIED: {occupied_count}"
    text_tot = f"TOTAL: {total_spaces} ({occ_pct:.1f}% Full)"

    cv2.putText(output, text_free, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FREE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(output, text_occ, (200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, OCCUPIED_COLOR, 2, cv2.LINE_AA)
    cv2.putText(output, text_tot, (440, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    return output


def make_pipeline_figure(
    patch_bgr: np.ndarray,
    ref_bgr: Optional[np.ndarray],
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Construct a step-by-step 6-panel educational pipeline progression image.
    
    Panels:
      1. Original Warped Slot Patch
      2. Grayscale Conversion
      3. CLAHE Contrast Equalization
      4. Canny Edge Map
      5. Background Difference Map (vs. reference)
      6. Final Decision Classification Badge
    """
    from src.preprocessing import to_grayscale, apply_clahe, apply_blur
    from src.features import edge_density, background_diff_score
    from src.classifier import rule_based_classify

    c_low = params.get("canny_low", 50)
    c_high = params.get("canny_high", 150)
    d_thresh = params.get("diff_threshold", 30)

    # 1. Original
    p1 = cv2.resize(patch_bgr, (120, 120))

    # 2. Grayscale
    gray = to_grayscale(patch_bgr)
    p2 = cv2.cvtColor(cv2.resize(gray, (120, 120)), cv2.COLOR_GRAY2BGR)

    # 3. CLAHE
    blurred = apply_blur(gray, params.get("blur_kernel", 5))
    clahe_img = apply_clahe(blurred)
    p3 = cv2.cvtColor(cv2.resize(clahe_img, (120, 120)), cv2.COLOR_GRAY2BGR)

    # 4. Canny Edges
    ratio, edges = edge_density(clahe_img, canny_low=c_low, canny_high=c_high)
    p4 = cv2.cvtColor(cv2.resize(edges, (120, 120)), cv2.COLOR_GRAY2BGR)

    # 5. Diff Map
    if ref_bgr is not None:
        ref_gray = apply_clahe(apply_blur(to_grayscale(ref_bgr)))
        diff_score, excess_frac, diff_map = background_diff_score(clahe_img, ref_gray, threshold=d_thresh)
    else:
        diff_map = np.zeros_like(clahe_img)
        diff_score = 0.0

    p5 = cv2.applyColorMap(cv2.resize(diff_map, (120, 120)), cv2.COLORMAP_JET)

    # 6. Final Decision Badge
    feat = {"edge_density": ratio, "diff_score": diff_score}
    pred, conf, score = rule_based_classify(
        feat,
        edge_weight=params.get("edge_weight", 0.5),
        diff_weight=params.get("diff_weight", 0.5),
        decision_threshold=params.get("decision_threshold", 0.35)
    )

    p6 = p1.copy()
    badge_color = OCCUPIED_COLOR if pred == 1 else FREE_COLOR
    cv2.rectangle(p6, (0, 0), (120, 120), badge_color, 4)
    status_str = "OCCUPIED" if pred == 1 else "FREE"
    cv2.putText(p6, status_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(p6, f"{int(conf*100)}%", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Assemble horizontally with titles
    panels = [p1, p2, p3, p4, p5, p6]
    titles = ["1. Original", "2. Grayscale", "3. CLAHE", "4. Canny Edges", "5. Diff Map", "6. Decision"]

    panel_h, panel_w = 120, 120
    header_h = 28
    composite = np.zeros((panel_h + header_h, panel_w * 6, 3), dtype=np.uint8)

    for i, (panel, title) in enumerate(zip(panels, titles)):
        x_start = i * panel_w
        composite[header_h:, x_start:x_start + panel_w] = panel
        cv2.putText(composite, title, (x_start + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        if i > 0:
            cv2.line(composite, (x_start, 0), (x_start, panel_h + header_h), (50, 50, 50), 1)

    return composite


def plot_confusion_matrix_heatmap(cm_data: Dict[str, int], save_path: Optional[Path] = None) -> plt.Figure:
    """Plot publication-quality 2x2 confusion matrix heatmap."""
    tp, tn = cm_data.get("TP", 0), cm_data.get("TN", 0)
    fp, fn = cm_data.get("FP", 0), cm_data.get("FN", 0)

    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)

    classes = ["Free (0)", "Occupied (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title("ParkLens Confusion Matrix (Test Split)", fontsize=12, fontweight="bold", pad=12)

    # Annotate numbers
    thresh = matrix.max() / 2.0
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            text = f"{names[i][j]}\n{val:,}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roc_curves(lot_rocs: Dict[str, Dict[str, Any]], save_path: Optional[Path] = None) -> plt.Figure:
    """Plot ROC curves with Area Under Curve (AUC) for each parking lot."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    colors = {"PUCPR": "#00BCD4", "UFPR04": "#FF9800", "UFPR05": "#4CAF50"}

    for lot, roc in lot_rocs.items():
        fpr = roc.get("fpr", [0, 1])
        tpr = roc.get("tpr", [0, 1])
        auc = roc.get("auc", 0.5)
        color = colors.get(lot, "#E91E63")
        ax.plot(fpr, tpr, label=f"{lot} (AUC = {auc:.3f})", color=color, linewidth=2)

    # Diagonal random guess line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curves by Parking Lot", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_weather_accuracy_bars(weather_metrics: Dict[str, Dict[str, float]], save_path: Optional[Path] = None) -> plt.Figure:
    """Bar chart comparing Accuracy, Recall, and Precision across Sunny, Cloudy, Rainy."""
    weathers = [w for w in ["Sunny", "Cloudy", "Rainy"] if w in weather_metrics]
    if not weathers:
        weathers = list(weather_metrics.keys())

    accs = [weather_metrics[w]["accuracy"] * 100 for w in weathers]
    recs = [weather_metrics[w]["recall"] * 100 for w in weathers]
    precs = [weather_metrics[w]["precision"] * 100 for w in weathers]

    x = np.arange(len(weathers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.bar(x - width, accs, width, label="Accuracy (%)", color="#00BCD4")
    ax.bar(x, recs, width, label="Recall / Sens (%)", color="#4CAF50")
    ax.bar(x + width, precs, width, label="Precision (%)", color="#FF9800")

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Across Weather Conditions", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(weathers, fontsize=10, fontweight="bold")
    ax.set_ylim([50, 105])
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def save_all_evaluation_figures(report: Dict[str, Any], output_dir: Path = OUTPUTS_DIR) -> None:
    """Render and save all standard figures to outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    cm_data = {
        "TP": report["overall"]["tp"],
        "TN": report["overall"]["tn"],
        "FP": report["overall"]["fp"],
        "FN": report["overall"]["fn"]
    }
    plot_confusion_matrix_heatmap(cm_data, output_dir / "confusion_matrix.png")

    # 2. ROC Curves
    if "lot_rocs" in report:
        plot_roc_curves(report["lot_rocs"], output_dir / "roc_curves.png")

    # 3. Weather Accuracy Bars
    if "by_weather" in report:
        plot_weather_accuracy_bars(report["by_weather"], output_dir / "weather_accuracy.png")

    print(f"Artifact plots saved to {output_dir}:")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curves.png")
    print(f"  - weather_accuracy.png")

