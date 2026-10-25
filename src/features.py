"""
src/features.py – Classical Computer Vision feature extractors for ParkLens.

Each feature extractor is implemented as a standalone function with heavy docstrings
explaining the underlying mathematical formulation and physical intuition for the student viva.

Features Extracted:
  1. Edge Density (Canny Edge Detection)
  2. Background Subtraction Difference (Mean absolute diff & pixel excess fraction)
  3. Laplacian Variance (Texture sharpness & frequency distribution)
  4. Grayscale Standard Deviation (Luminance dispersion)
  5. Color Histogram Distance (HSV Chi-Squared distance)
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

from config import (
    CANNY_LOW, CANNY_HIGH, DIFF_THRESHOLD,
    BLUR_KERNEL, CLAHE_CLIP, CLAHE_GRID
)
from src.preprocessing import preprocess_patch


def edge_density(
    gray_patch: np.ndarray,
    canny_low: int = CANNY_LOW,
    canny_high: int = CANNY_HIGH,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Compute edge density using the Canny edge detector.
    
    Physical / Computer Vision Intuition:
      Empty parking spots consist of relatively flat, textureless asphalt or paving stones
      with at most a painted lane marking. When a vehicle occupies the stall, it introduces
      numerous sharp geometric discontinuities: car hood seams, side panels, windshield
      frames, radiator grilles, license plates, and wheels. Consequently, occupied stalls
      exhibit a dramatically higher fraction of edge pixels than vacant stalls.
      
    Math:
      Edge Density = (Count of non-zero edge pixels in Canny map) / (Total pixel count in mask)
      
    Args:
        gray_patch: Single-channel grayscale image (preferably after CLAHE/blur).
        canny_low: Lower hysteresis threshold for Canny.
        canny_high: Upper hysteresis threshold for Canny.
        mask: Optional binary mask. If None, considers the entire patch area.
        
    Returns:
        (edge_ratio, edge_image): Float in [0.0, 1.0] and uint8 edge map.
    """
    edges = cv2.Canny(gray_patch, canny_low, canny_high)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        total_pixels = max(1, int(np.count_nonzero(mask)))
    else:
        total_pixels = gray_patch.shape[0] * gray_patch.shape[1]

    edge_pixels = int(np.count_nonzero(edges))
    ratio = float(edge_pixels) / float(total_pixels)
    return float(np.clip(ratio, 0.0, 1.0)), edges


def background_diff_score(
    patch_gray: np.ndarray,
    ref_gray: np.ndarray,
    threshold: int = DIFF_THRESHOLD
) -> Tuple[float, float, np.ndarray]:
    """
    Compute pixel difference score against the empty slot reference patch.
    
    Physical / Computer Vision Intuition:
      Classical background subtraction relies on comparing the observation against an
      established stationary reference model. When a slot is vacant, the current patch
      closely resembles the reference pavement patch (save for slight illumination drift,
      which CLAHE neutralizes). An intruding vehicle produces significant intensity
      discrepancies across large contiguous regions.
      
    Math:
      Difference map D(x, y) = |I_current(x, y) - I_reference(x, y)|
      Mean Difference = (1 / N) * sum(D(x, y)) / 255.0
      Excess Fraction = (Count of pixels where D(x, y) > threshold) / N
      Combined Diff Score = 0.5 * Mean Difference + 0.5 * Excess Fraction
      
    Args:
        patch_gray: Preprocessed grayscale patch of current slot.
        ref_gray: Preprocessed grayscale patch of canonical empty slot.
        threshold: Intensity difference threshold to classify a pixel as "changed".
        
    Returns:
        (diff_score, excess_fraction, diff_map): Normalized scores and absolute difference image.
    """
    if ref_gray is None or patch_gray.shape != ref_gray.shape:
        # Fallback if no reference exists
        return 0.0, 0.0, np.zeros_like(patch_gray)

    diff_map = cv2.absdiff(patch_gray, ref_gray)
    mean_diff = float(np.mean(diff_map)) / 255.0
    excess_pixels = int(np.count_nonzero(diff_map > threshold))
    total_pixels = diff_map.size
    excess_fraction = float(excess_pixels) / float(total_pixels)

    # Combined normalized difference score in [0.0, 1.0]
    diff_score = 0.5 * mean_diff + 0.5 * excess_fraction
    return float(np.clip(diff_score, 0.0, 1.0)), excess_fraction, diff_map


def laplacian_variance(gray_patch: np.ndarray) -> float:
    """
    Compute the variance of the Laplacian of the grayscale patch.
    
    Physical / Computer Vision Intuition:
      The discrete Laplacian operator approximates the second spatial derivative of the image,
      measuring the rate of gradient change (high spatial frequencies). Empty asphalt has low
      variance because spatial transitions are gradual or noise-dominated. Cars introduce
      distinct, sharp metallic borders and textures, spreading the second derivative distribution
      and yielding significantly higher Laplacian variance.
      
    Math:
      L(x, y) = (d^2 I / dx^2) + (d^2 I / dy^2)
      Score = Variance(L(x, y))
    """
    lap = cv2.Laplacian(gray_patch, cv2.CV_64F)
    score = float(np.var(lap))
    return score


def gray_std(gray_patch: np.ndarray) -> float:
    """
    Compute the standard deviation of pixel intensities in the grayscale patch.
    
    Physical Intuition:
      Measures global intensity spread. Vacant asphalt has a narrow histogram centered
      around mid-gray tones. Vehicles contain deep darks (tires, tinted windows) and bright
      highlights (car paint reflections, chrome), resulting in a broader standard deviation.
    """
    return float(np.std(gray_patch))


def color_histogram_diff(
    patch_hsv: Optional[np.ndarray],
    ref_hsv: Optional[np.ndarray],
    bins: Tuple[int, int] = (16, 16)
) -> float:
    """
    Compute Chi-Squared distance between 2D Hue-Saturation color histograms.
    
    Physical Intuition:
      Vehicles introduce vibrant chromatic signatures (reds, blues, yellows, whites)
      that sharply contrast with the achromatic/grayish signature of bare asphalt.
    """
    if patch_hsv is None or ref_hsv is None:
        return 0.0

    h_bins, s_bins = bins
    hist_patch = cv2.calcHist([patch_hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    hist_ref = cv2.calcHist([ref_hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])

    cv2.normalize(hist_patch, hist_patch, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(hist_ref, hist_ref, alpha=1.0, norm_type=cv2.NORM_L1)

    # Chi-Squared distance
    dist = cv2.compareHist(hist_patch, hist_ref, cv2.HISTCMP_CHISQR)
    return float(np.clip(dist, 0.0, 10.0))


def extract_features(
    patch_bgr: np.ndarray,
    ref_patch_bgr: Optional[np.ndarray] = None,
    canny_low: int = CANNY_LOW,
    canny_high: int = CANNY_HIGH,
    diff_threshold: int = DIFF_THRESHOLD,
    clahe_clip: float = CLAHE_CLIP,
    clahe_grid: Tuple[int, int] = CLAHE_GRID,
    blur_k: int = BLUR_KERNEL
) -> Dict[str, Any]:
    """
    Complete feature extraction pipeline on a parking space patch.
    
    Preprocesses the patch, extracts edge, background difference, and texture features,
    and returns a structured dictionary along with intermediate visual maps for UI display.
    """
    prep = preprocess_patch(patch_bgr, blur_k, clahe_clip, clahe_grid)
    clahe_gray = prep["clahe"]
    patch_hsv = prep["hsv"]

    # Preprocess reference patch if provided
    ref_clahe_gray = None
    ref_hsv = None
    if ref_patch_bgr is not None:
        ref_prep = preprocess_patch(ref_patch_bgr, blur_k, clahe_clip, clahe_grid)
        ref_clahe_gray = ref_prep["clahe"]
        ref_hsv = ref_prep["hsv"]

    # 1. Edge Density
    edge_ratio, edge_map = edge_density(clahe_gray, canny_low, canny_high)

    # 2. Background Subtraction
    diff_score, excess_frac, diff_map = background_diff_score(
        clahe_gray, ref_clahe_gray, threshold=diff_threshold
    )

    # 3. Texture / Variance
    lap_var = laplacian_variance(clahe_gray)
    std_val = gray_std(clahe_gray)

    # 4. Color distance
    color_dist = color_histogram_diff(patch_hsv, ref_hsv)

    return {
        "edge_density": edge_ratio,
        "diff_score": diff_score,
        "diff_excess_fraction": excess_frac,
        "laplacian_variance": lap_var,
        "gray_std": std_val,
        "color_dist": color_dist,
        "edge_map": edge_map,
        "diff_map": diff_map,
        "preprocessed": prep
    }


def features_to_vector(feat_dict: Dict[str, Any]) -> np.ndarray:
    """
    Flatten core numerical features into a 1D vector suitable for scikit-learn models.
    """
    return np.array([
        feat_dict["edge_density"],
        feat_dict["diff_score"],
        feat_dict["diff_excess_fraction"],
        feat_dict["laplacian_variance"] / 1000.0,  # scale for numerical conditioning
        feat_dict["gray_std"] / 100.0,
        feat_dict["color_dist"]
    ], dtype=np.float32)
