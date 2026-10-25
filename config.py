"""
config.py – Central configuration for the ParkLens project.

All tuneable constants live here so every module can import them.
Paths are relative to the project root.
"""

import os
from pathlib import Path

# ─── Project root (directory containing this file) ──────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ─── Dataset paths ──────────────────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
PKLOT_DIR: Path = DATA_DIR / "PKLot"
INDEX_FILE: Path = DATA_DIR / "index.json"
MANIFEST_FILE: Path = DATA_DIR / "MANIFEST.json"

# ─── Output paths ───────────────────────────────────────────────────────────
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
PARAMS_FILE: Path = OUTPUTS_DIR / "params.json"
RESULTS_FILE: Path = OUTPUTS_DIR / "results.json"
RESULTS_CSV: Path = OUTPUTS_DIR / "results.csv"
METRICS_FILE: Path = OUTPUTS_DIR / "metrics.json"

# ─── Download configuration ─────────────────────────────────────────────────
# Target number of images per lot × weather group.
# Reduce if bandwidth is poor (minimum ~50).
IMAGES_PER_GROUP: int = 100

# PKLot parking lots
LOTS: list[str] = ["PUCPR", "UFPR04", "UFPR05"]

# Weather conditions in PKLot
WEATHERS: list[str] = ["Sunny", "Cloudy", "Rainy"]

# Download retry settings
DOWNLOAD_TIMEOUT: int = 30        # seconds per request
DOWNLOAD_RETRIES: int = 3         # number of retries per file
DOWNLOAD_CHUNK_SIZE: int = 8192   # bytes

# ─── Train / test split ─────────────────────────────────────────────────────
TRAIN_RATIO: float = 0.80
RANDOM_SEED: int = 42

# ─── Image processing defaults ──────────────────────────────────────────────
PATCH_SIZE: tuple[int, int] = (64, 64)       # warped space patch size (w, h)
BLUR_KERNEL: int = 5                         # Gaussian blur kernel size
CLAHE_CLIP: float = 2.0                      # CLAHE clip limit
CLAHE_GRID: tuple[int, int] = (8, 8)         # CLAHE tile grid size

# ─── Feature / classifier defaults ──────────────────────────────────────────
CANNY_LOW: int = 50                          # Canny lower threshold
CANNY_HIGH: int = 150                        # Canny upper threshold
DIFF_THRESHOLD: int = 30                     # abs-diff binary threshold
EDGE_WEIGHT: float = 0.5                     # weight for edge density in score
DIFF_WEIGHT: float = 0.5                     # weight for diff score
DECISION_THRESHOLD: float = 0.35             # global decision threshold

# ─── Grid search ranges (for tuning) ────────────────────────────────────────
TUNE_CANNY_LOW_RANGE: list[int] = [30, 50, 70]
TUNE_CANNY_HIGH_RANGE: list[int] = [100, 150, 200]
TUNE_DIFF_THRESH_RANGE: list[int] = [20, 30, 40, 50]
TUNE_DECISION_THRESH_RANGE: list[float] = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# ─── Visualization ──────────────────────────────────────────────────────────
FREE_COLOR: tuple[int, int, int] = (0, 255, 0)       # BGR green
OCCUPIED_COLOR: tuple[int, int, int] = (0, 0, 255)   # BGR red
ERROR_COLOR: tuple[int, int, int] = (0, 255, 255)    # BGR yellow
FONT_SCALE: float = 0.5
LINE_THICKNESS: int = 2

# ─── Streamlit / UI ─────────────────────────────────────────────────────────
TIMELINE_MAX_FRAMES: int = 30   # cap for the timeline tab

# ─── Matplotlib config dir workaround ───────────────────────────────────────
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))

