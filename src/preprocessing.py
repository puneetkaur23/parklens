"""
src/preprocessing.py – Classical image processing primitives for ParkLens.

Pipeline Stages for Every Space:
  1. Warp: Perspective warp quadrilateral slot contour to canonical (64x64) patch.
  2. Grayscale: Eliminate chromatic variance; preserve luminance.
  3. Gaussian Blur: Suppress high-frequency sensor noise (e.g. 5x5 kernel, sigma=0).
  4. CLAHE: Contrast Limited Adaptive Histogram Equalization to handle dramatic
     lighting changes (shadows, direct sun specularities, cloudy overcast).
  5. HSV Conversion: Extract Hue and Saturation channels for optional color divergence.

Every function is pure, modular, heavily commented, and exposes intermediate states
so that the Streamlit UI can render the complete visual journey.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from config import PATCH_SIZE, BLUR_KERNEL, CLAHE_CLIP, CLAHE_GRID
from src.utils import warp_polygon_to_patch


def extract_space_patch(
    image: np.ndarray,
    polygon: List[List[int]],
    patch_size: Tuple[int, int] = PATCH_SIZE
) -> np.ndarray:
    """
    Extract and homography-rectify a single parking space into a fixed-resolution patch.
    
    Args:
        image: Original full-lot BGR image.
        polygon: List of [x, y] coordinates defining the slot boundary.
        patch_size: Target (width, height), default (64, 64).
        
    Returns:
        Warped BGR image patch.
    """
    return warp_polygon_to_patch(image, polygon, patch_size)


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a 3-channel BGR image to single-channel 8-bit Grayscale.
    
    Uses standard OpenCV luminance weights: Y = 0.299*R + 0.587*G + 0.114*B.
    """
    if len(image_bgr.shape) == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def apply_blur(gray: np.ndarray, kernel_size: int = BLUR_KERNEL) -> np.ndarray:
    """
    Apply 2D Gaussian smoothing filter to suppress high-frequency asphalt noise.
    
    Mathematical formula:
      G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    Kernel size must be an odd positive integer (e.g. 3, 5, 7).
    """
    if kernel_size <= 1:
        return gray
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = CLAHE_CLIP,
    grid_size: Tuple[int, int] = CLAHE_GRID
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Why CLAHE over standard Global Histogram Equalization?
      Global equalization tends to over-amplify noise and blows out bright specular highlights
      on sunny days or washes out shadows. CLAHE operates on localized contextual tiles
      (e.g., 8x8 pixels) and clips the local histogram at clip_limit before redistributing
      it, preventing excessive noise amplification while enhancing subtle structural contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray)


def apply_histogram_eq(gray: np.ndarray) -> np.ndarray:
    """
    Apply standard global histogram equalization for comparison.
    """
    return cv2.equalizeHist(gray)


def to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to HSV (Hue, Saturation, Value) color space.
    Decouples chromatic information (Hue) from intensity (Value).
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def preprocess_patch(
    patch_bgr: np.ndarray,
    blur_k: int = BLUR_KERNEL,
    clahe_clip: float = CLAHE_CLIP,
    clahe_grid: Tuple[int, int] = CLAHE_GRID
) -> Dict[str, np.ndarray]:
    """
    Run full preprocessing pipeline on a parking space patch and return all intermediate states.
    
    Returns:
      Dictionary containing:
        - 'bgr': Original patch
        - 'gray': Grayscale representation
        - 'blurred': After Gaussian noise reduction
        - 'clahe': After localized contrast enhancement
        - 'hsv': HSV color representation
    """
    gray = to_grayscale(patch_bgr)
    blurred = apply_blur(gray, kernel_size=blur_k)
    clahe_out = apply_clahe(blurred, clip_limit=clahe_clip, grid_size=clahe_grid)
    hsv_out = to_hsv(patch_bgr) if len(patch_bgr.shape) == 3 else None

    return {
        "bgr": patch_bgr,
        "gray": gray,
        "blurred": blurred,
        "clahe": clahe_out,
        "hsv": hsv_out
    }
