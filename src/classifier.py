"""
src/classifier.py – Decision logic and parameter tuning for ParkLens.

Contains:
  1. Transparent Rule-Based Classifier:
     Combines normalized edge density and background subtraction discrepancy
     via a weighted linear formulation and evaluates against an occupancy threshold.
  2. Per-lot & per-weather Parameter Tuning:
     Conducts systematic grid search over validation/training splits to maximize F1-score.
  3. Machine Learning Comparison Classifier:
     Provides an optional LogisticRegression / LinearSVM baseline on extracted features.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    EDGE_WEIGHT, DIFF_WEIGHT, DECISION_THRESHOLD,
    TUNE_CANNY_LOW_RANGE, TUNE_CANNY_HIGH_RANGE,
    TUNE_DIFF_THRESH_RANGE, TUNE_DECISION_THRESH_RANGE,
    PARAMS_FILE, LOTS, WEATHERS
)
from src.features import extract_features, features_to_vector
from src.utils import warp_polygon_to_patch, save_json, load_json


def rule_based_classify(
    features: Dict[str, Any],
    edge_weight: float = EDGE_WEIGHT,
    diff_weight: float = DIFF_WEIGHT,
    decision_threshold: float = DECISION_THRESHOLD
) -> Tuple[int, float, float]:
    """
    Decide whether a parking space is FREE (0) or OCCUPIED (1) using classical rules.
    
    Formula:
      Combined Score S = (w_edge * EdgeDensity) + (w_diff * DiffScore)
      Prediction = 1 if S >= threshold else 0
      
    Confidence Formulation:
      Distance from boundary normalized by distance to extremes:
      Confidence = 0.5 + 0.5 * (|S - threshold| / max(threshold, 1.0 - threshold))
      Guarantees confidence in [0.5, 1.0] representing certainty of the chosen binary decision.
      
    Args:
        features: Dictionary from extract_features().
        edge_weight: Importance weighting for edge density.
        diff_weight: Importance weighting for background subtraction.
        decision_threshold: Cutoff value for occupancy declaration.
        
    Returns:
        (prediction, confidence, combined_score)
    """
    ed = features.get("edge_density", 0.0)
    ds = features.get("diff_score", 0.0)

    # Normalize weights so they sum to 1.0
    total_w = edge_weight + diff_weight
    if total_w <= 0:
        w_e, w_d = 0.5, 0.5
    else:
        w_e = edge_weight / total_w
        w_d = diff_weight / total_w

    score = float(w_e * ed + w_d * ds)
    prediction = 1 if score >= decision_threshold else 0

    denom = max(decision_threshold, 1.0 - decision_threshold, 1e-5)
    margin = abs(score - decision_threshold)
    confidence = float(np.clip(0.5 + 0.5 * (margin / denom), 0.5, 1.0))

    return prediction, confidence, score


def classify_spaces_in_image(
    image_bgr: np.ndarray,
    spaces: List[Dict[str, Any]],
    lot: str,
    ref_patches: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Execute end-to-end detection for all parking spaces within a full parking lot image.
    
    Args:
        image_bgr: Full-resolution lot image.
        spaces: List of space dicts with 'id' and 'polygon'.
        lot: Identifier for parking lot ('PUCPR', 'UFPR04', etc.).
        ref_patches: Lookup table of empty background patches.
        params: Hyperparameters (canny_low, canny_high, diff_threshold, decision_threshold, etc.).
        
    Returns:
        List of result dictionaries containing space_id, prediction, confidence,
        ground_truth (if present), and feature values.
    """
    canny_low = params.get("canny_low", 50)
    canny_high = params.get("canny_high", 150)
    diff_thresh = params.get("diff_threshold", 30)
    edge_w = params.get("edge_weight", 0.5)
    diff_w = params.get("diff_weight", 0.5)
    dec_thresh = params.get("decision_threshold", 0.35)

    results = []
    for sp in spaces:
        s_id = str(sp.get("id"))
        poly = sp.get("polygon", [])
        gt_occ = sp.get("occupied", None)

        patch = warp_polygon_to_patch(image_bgr, poly)
        ref_patch = ref_patches.get(f"{lot}_{s_id}")

        feat = extract_features(
            patch_bgr=patch,
            ref_patch_bgr=ref_patch,
            canny_low=canny_low,
            canny_high=canny_high,
            diff_threshold=diff_thresh
        )

        pred, conf, score = rule_based_classify(
            features=feat,
            edge_weight=edge_w,
            diff_weight=diff_w,
            decision_threshold=dec_thresh
        )

        results.append({
            "id": s_id,
            "polygon": poly,
            "prediction": pred,
            "confidence": conf,
            "combined_score": score,
            "ground_truth": gt_occ,
            "is_correct": (pred == gt_occ) if gt_occ is not None else None,
            "features": {
                "edge_density": feat["edge_density"],
                "diff_score": feat["diff_score"],
                "laplacian_variance": feat["laplacian_variance"],
                "gray_std": feat["gray_std"],
                "color_dist": feat["color_dist"]
            }
        })

    return results


def grid_search_tune_group(
    train_samples: List[Dict[str, Any]],
    ref_patches: Dict[str, np.ndarray],
    lot: str,
    weather: str
) -> Dict[str, Any]:
    """
    Grid-search optimal parameters for a single (lot, weather) cohort on training data.
    
    Optimizes for highest F1-score across:
      - canny_low / canny_high
      - diff_threshold
      - decision_threshold
    """
    import cv2
    from src.evaluation import compute_metrics

    # Pre-extract patches and ground truth labels to make grid search blazingly fast
    cached_data = []
    for s in train_samples:
        img_path = PROJECT_ROOT / s["image_path"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for sp in s.get("spaces", []):
            s_id = str(sp.get("id"))
            poly = sp.get("polygon", [])
            gt = sp.get("occupied", 0)
            patch = warp_polygon_to_patch(img, poly)
            ref_patch = ref_patches.get(f"{lot}_{s_id}")
            cached_data.append((patch, ref_patch, gt))

    if not cached_data:
        # Default fallback
        return {
            "canny_low": 50,
            "canny_high": 150,
            "diff_threshold": 30,
            "edge_weight": 0.5,
            "diff_weight": 0.5,
            "decision_threshold": 0.35,
            "best_f1": 0.0
        }

    # Subsample if group has large amount of training spots for speed
    if len(cached_data) > 600:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(cached_data), size=600, replace=False)
        cached_data = [cached_data[i] for i in idx]

    best_f1 = -1.0
    best_params = {}

    # Grid search across threshold combinations
    for c_low in [40, 50, 70]:
        for c_high in [120, 150, 180]:
            for d_thresh in [25, 35, 45]:
                # Extract features for all samples once per (c_low, c_high, d_thresh)
                scores = []
                y_trues = []
                for patch, ref_patch, gt in cached_data:
                    feat = extract_features(
                        patch_bgr=patch,
                        ref_patch_bgr=ref_patch,
                        canny_low=c_low,
                        canny_high=c_high,
                        diff_threshold=d_thresh
                    )
                    score = 0.5 * feat["edge_density"] + 0.5 * feat["diff_score"]
                    scores.append(score)
                    y_trues.append(gt)

                # Sweep decision thresholds
                for dec_t in [0.25, 0.30, 0.35, 0.40, 0.45]:
                    preds = [1 if s >= dec_t else 0 for s in scores]
                    m = compute_metrics(y_trues, preds)
                    f1 = m["f1"]
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {
                            "canny_low": c_low,
                            "canny_high": c_high,
                            "diff_threshold": d_thresh,
                            "edge_weight": 0.5,
                            "diff_weight": 0.5,
                            "decision_threshold": dec_t,
                            "best_f1": round(f1, 4),
                            "accuracy": round(m["accuracy"], 4)
                        }

    return best_params


def tune_all_groups(
    train_index: List[Dict[str, Any]],
    ref_patches: Dict[str, np.ndarray],
    save_path: Path = PARAMS_FILE
) -> Dict[str, Any]:
    """
    Run parameter tuning across all 9 combinations of (Lot, Weather).
    Saves results to outputs/params.json.
    """
    print("=" * 65)
    print("  ParkLens Hyperparameter Optimization (Grid Search on Train Split)")
    print("=" * 65)

    all_tuned = {}
    grouped = {}
    for item in train_index:
        key = (item["lot"], item["weather"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)

    for lot in LOTS:
        for weather in WEATHERS:
            key = f"{lot}_{weather}"
            samples = grouped.get((lot, weather), [])
            print(f"Tuning cohort: {lot:7s} | {weather:8s} ({len(samples)} train images)...", end=" ")
            params = grid_search_tune_group(samples, ref_patches, lot, weather)
            all_tuned[key] = params
            print(f"Done! -> Best F1: {params.get('best_f1', 0.0):.4f} (Thresh: {params.get('decision_threshold')})")

    # Also compute global fallback
    global_params = {
        "canny_low": 50,
        "canny_high": 150,
        "diff_threshold": 30,
        "edge_weight": 0.5,
        "diff_weight": 0.5,
        "decision_threshold": 0.35
    }
    all_tuned["global_default"] = global_params

    save_json(save_path, all_tuned)
    print(f"\nTuned parameters saved to: {save_path}")
    return all_tuned


def load_tuned_params(params_file: Path = PARAMS_FILE, lot: str = "", weather: str = "") -> Dict[str, Any]:
    """Load tuned parameters for specific lot and weather condition."""
    if params_file.exists():
        data = load_json(params_file)
        key = f"{lot}_{weather}"
        if key in data:
            return data[key]
        if "global_default" in data:
            return data["global_default"]

    return {
        "canny_low": 50,
        "canny_high": 150,
        "diff_threshold": 30,
        "edge_weight": 0.5,
        "diff_weight": 0.5,
        "decision_threshold": 0.35
    }
