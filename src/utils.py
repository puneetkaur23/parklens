"""
src/utils.py – Utility functions for geometry, I/O, and image transformations.

Contains:
  - Seed initialization for reproducible results
  - Bounding box and polygon conversions
  - Polygon perspective warping to canonical patches
  - Metric file and JSON I/O helpers
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across random and numpy for strict repeatability."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    """Load and parse JSON file safely with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Save serializable data to JSON file with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order coordinates of 4 points in top-left, top-right, bottom-right, bottom-left order.
    
    Required for consistent homography / perspective warping.
    Top-left has the smallest sum (x + y), bottom-right has largest sum.
    Top-right has smallest difference (y - x), bottom-left has largest difference.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_polygon_to_patch(
    image: np.ndarray,
    polygon: List[List[int]],
    target_size: Tuple[int, int] = (64, 64)
) -> np.ndarray:
    """
    Warp an arbitrary quadrilateral/polygon region in an image into a normalized patch.
    
    If polygon has exactly 4 points, uses perspective transform (homography).
    If polygon has != 4 points, computes minimum area bounding box or bounding rect
    and warps the 4 box corners.
    
    Args:
        image: Source image (BGR or Grayscale).
        polygon: List of [x, y] coordinates.
        target_size: (width, height) tuple for destination patch.
        
    Returns:
        Warped patch as np.ndarray with shape (target_size[1], target_size[0], ...).
    """
    pts = np.array(polygon, dtype=np.float32)
    tw, th = target_size

    if len(pts) == 4:
        rect = order_points(pts)
    else:
        # Fit rotated rectangle to multi-point contour
        rect_box = cv2.minAreaRect(pts.astype(np.int32))
        box_pts = cv2.boxPoints(rect_box)
        rect = order_points(box_pts)

    dst = np.array([
        [0, 0],
        [tw - 1, 0],
        [tw - 1, th - 1],
        [0, th - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


def polygon_to_mask(polygon: List[List[int]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Generate binary boolean mask (0 and 255) for polygon within an image shape (height, width).
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask
