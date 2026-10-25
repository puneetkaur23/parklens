"""
src/evaluation.py – Comprehensive quantitative metrics and evaluation on the TEST split.

Metrics Computed:
  - Confusion Matrix: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
  - Accuracy = (TP + TN) / (TP + TN + FP + FN)
  - Sensitivity (Recall) = TP / (TP + FN)  [Ability to detect occupied cars]
  - Specificity = TN / (TN + FP)           [Ability to declare empty spots free]
  - Precision = TP / (TP + FP)             [Fraction of occupied declarations that are true cars]
  - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
  - ROC Curve (FPR vs TPR across thresholds) and Area Under Curve (AUC)
  
Generates both per-lot and per-lot×weather results tables saved to outputs/ as JSON and CSV.
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUTS_DIR, RESULTS_FILE, RESULTS_CSV, METRICS_FILE, LOTS, WEATHERS
from src.classifier import classify_spaces_in_image, load_tuned_params
from src.utils import save_json, load_json


def confusion_matrix_data(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Compute counts for TP, TN, FP, FN."""
    tp, tn, fp, fn = 0, 0, 0, 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute classical diagnostic performance metrics.
    
    Sensitivity (Recall) measures the fraction of occupied cars correctly caught.
    Specificity measures the reliability of identifying free spaces for drivers.
    Precision measures how trustworthy an 'occupied' alert is.
    F1-Score is the harmonic mean of Precision and Sensitivity.
    """
    cm = confusion_matrix_data(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        "accuracy": float(round(accuracy, 4)),
        "sensitivity": float(round(sensitivity, 4)),
        "recall": float(round(sensitivity, 4)),
        "specificity": float(round(specificity, 4)),
        "precision": float(round(precision, 4)),
        "f1": float(round(f1, 4)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total
    }


def compute_roc(y_true: List[int], y_scores: List[float], num_thresholds: int = 100) -> Dict[str, Any]:
    """
    Compute Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC).
    
    Sweeps decision threshold from 0.0 to 1.0, tracking:
      False Positive Rate (1 - Specificity) vs True Positive Rate (Sensitivity).
      AUC is numerically integrated using the trapezoidal rule.
    """
    if not y_true or not y_scores or len(y_true) != len(y_scores):
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0], "auc": 0.5}

    y_t = np.array(y_true)
    y_s = np.array(y_scores)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    # Sort thresholds descending from 1.0 to 0.0 for standard ROC
    thresholds = np.sort(thresholds)[::-1]

    n_pos = np.sum(y_t == 1)
    n_neg = np.sum(y_t == 0)

    if n_pos == 0 or n_neg == 0:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0], "auc": 0.5}

    fpr_list = []
    tpr_list = []

    for t in thresholds:
        preds = (y_s >= t).astype(int)
        tp = np.sum((preds == 1) & (y_t == 1))
        fp = np.sum((preds == 1) & (y_t == 0))
        tpr = tp / n_pos
        fpr = fp / n_neg
        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))

    # Numerical integration using trapezoidal rule or sklearn
    try:
        if hasattr(np, "trapezoid"):
            auc_val = float(np.trapezoid(tpr_list, fpr_list))
        elif hasattr(np, "trapz"):
            auc_val = float(np.trapz(tpr_list, fpr_list))
        else:
            auc_val = sum(0.5 * (tpr_list[i] + tpr_list[i+1]) * abs(fpr_list[i+1] - fpr_list[i]) for i in range(len(tpr_list)-1))
    except Exception:
        auc_val = 0.5
    auc_val = abs(auc_val)

    return {
        "fpr": fpr_list,
        "tpr": tpr_list,
        "thresholds": [float(t) for t in thresholds],
        "auc": float(round(auc_val, 4))
    }


def evaluate_test_set(
    test_index: List[Dict[str, Any]],
    ref_patches: Dict[str, np.ndarray],
    params_file: Path
) -> Dict[str, Any]:
    """
    Execute full evaluation of ParkLens on the unseen TEST split only.
    
    Generates:
      - Global overall metrics
      - Breakdown per parking lot (PUCPR, UFPR04, UFPR05)
      - Breakdown per weather condition (Sunny, Cloudy, Rainy)
      - Breakdown per lot × weather combination
      - ROC curves and AUC per parking lot
      - List of all false positive and false negative failure instances
    """
    import cv2
    from tqdm import tqdm

    print("=" * 65)
    print("  ParkLens Comprehensive Evaluation on Unseen Test Split")
    print(f"  Test images count: {len(test_index)}")
    print("=" * 65)

    all_y_true = []
    all_y_pred = []
    all_scores = []

    by_lot = {lot: {"true": [], "pred": [], "scores": []} for lot in LOTS}
    by_weather = {w: {"true": [], "pred": []} for w in WEATHERS}
    by_lot_weather = {}

    failure_cases = []

    for item in tqdm(test_index, desc="Evaluating Test Set", unit="img"):
        img_path = PROJECT_ROOT / item["image_path"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        lot = item.get("lot", "")
        weather = item.get("weather", "")
        params = load_tuned_params(params_file, lot=lot, weather=weather)

        spaces = item.get("spaces", [])
        results = classify_spaces_in_image(img, spaces, lot, ref_patches, params)

        lw_key = f"{lot}_{weather}"
        if lw_key not in by_lot_weather:
            by_lot_weather[lw_key] = {"true": [], "pred": []}

        for r in results:
            gt = r.get("ground_truth")
            if gt is None:
                continue

            pred = r["prediction"]
            score = r["combined_score"]

            all_y_true.append(gt)
            all_y_pred.append(pred)
            all_scores.append(score)

            if lot in by_lot:
                by_lot[lot]["true"].append(gt)
                by_lot[lot]["pred"].append(pred)
                by_lot[lot]["scores"].append(score)

            if weather in by_weather:
                by_weather[weather]["true"].append(gt)
                by_weather[weather]["pred"].append(pred)

            by_lot_weather[lw_key]["true"].append(gt)
            by_lot_weather[lw_key]["pred"].append(pred)

            # Record failure case
            if pred != gt:
                failure_cases.append({
                    "image_path": item["image_path"],
                    "lot": lot,
                    "weather": weather,
                    "datetime_str": item.get("datetime_str", ""),
                    "space_id": r["id"],
                    "ground_truth": gt,
                    "prediction": pred,
                    "error_type": "False Positive" if pred == 1 else "False Negative",
                    "confidence": r["confidence"],
                    "combined_score": round(score, 4),
                    "features": r["features"]
                })

    # Compute overall metrics
    overall = compute_metrics(all_y_true, all_y_pred)
    overall_roc = compute_roc(all_y_true, all_scores)
    overall["auc"] = overall_roc["auc"]

    # Compute per-lot metrics and ROCs
    lot_metrics = {}
    lot_rocs = {}
    for lot, d in by_lot.items():
        if d["true"]:
            m = compute_metrics(d["true"], d["pred"])
            roc = compute_roc(d["true"], d["scores"])
            m["auc"] = roc["auc"]
            lot_metrics[lot] = m
            lot_rocs[lot] = roc

    # Compute per-weather metrics
    weather_metrics = {}
    for w, d in by_weather.items():
        if d["true"]:
            weather_metrics[w] = compute_metrics(d["true"], d["pred"])

    # Compute per lot × weather metrics
    lot_weather_metrics = {}
    for lw, d in by_lot_weather.items():
        if d["true"]:
            lot_weather_metrics[lw] = compute_metrics(d["true"], d["pred"])

    evaluation_report = {
        "overall": overall,
        "overall_roc": overall_roc,
        "by_lot": lot_metrics,
        "by_weather": weather_metrics,
        "by_lot_weather": lot_weather_metrics,
        "lot_rocs": lot_rocs,
        "failure_cases_summary": {
            "total_failures": len(failure_cases),
            "false_positives": sum(1 for f in failure_cases if f["error_type"] == "False Positive"),
            "false_negatives": sum(1 for f in failure_cases if f["error_type"] == "False Negative")
        },
        "sample_failures": failure_cases[:30]
    }

    return evaluation_report


def save_evaluation_results(report: Dict[str, Any], output_dir: Path = OUTPUTS_DIR) -> None:
    """Save metrics to JSON and CSV formats in outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full JSON
    json_path = output_dir / "results.json"
    save_json(json_path, report)

    # 2. Save flat metrics summary JSON
    metrics_path = output_dir / "metrics.json"
    flat_metrics = {
        "overall": report["overall"],
        "by_lot": report["by_lot"],
        "by_weather": report["by_weather"],
        "by_lot_weather": report["by_lot_weather"]
    }
    save_json(metrics_path, flat_metrics)

    # 3. Save CSV table of lot x weather results
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cohort", "Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1-Score", "Total Spaces", "TP", "TN", "FP", "FN"])
        
        # Overall
        o = report["overall"]
        writer.writerow(["OVERALL", o["accuracy"], o["sensitivity"], o["specificity"], o["precision"], o["f1"], o["total"], o["tp"], o["tn"], o["fp"], o["fn"]])

        # By Lot
        for lot, m in report.get("by_lot", {}).items():
            writer.writerow([f"Lot: {lot}", m["accuracy"], m["sensitivity"], m["specificity"], m["precision"], m["f1"], m["total"], m["tp"], m["tn"], m["fp"], m["fn"]])

        # By Weather
        for w, m in report.get("by_weather", {}).items():
            writer.writerow([f"Weather: {w}", m["accuracy"], m["sensitivity"], m["specificity"], m["precision"], m["f1"], m["total"], m["tp"], m["tn"], m["fp"], m["fn"]])

        # By Lot x Weather
        for lw, m in sorted(report.get("by_lot_weather", {}).items()):
            writer.writerow([lw, m["accuracy"], m["sensitivity"], m["specificity"], m["precision"], m["f1"], m["total"], m["tp"], m["tn"], m["fp"], m["fn"]])

    print(f"\nEvaluation outputs saved:")
    print(f"  - {json_path}")
    print(f"  - {metrics_path}")
    print(f"  - {csv_path}")
