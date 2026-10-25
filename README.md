# ParkLens: Classical Computer Vision Parking Occupancy Detector

[![Course: Fundamentals of Computer Vision](https://img.shields.io/badge/Course-Fundamentals%20of%20CV-00BCD4.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Framework: OpenCV + NumPy](https://img.shields.io/badge/Stack-OpenCV%20%7C%20NumPy%20%7C%20Streamlit-brightgreen.svg)](#)
[![Dataset: PKLot CC BY 4.0](https://img.shields.io/badge/Dataset-PKLot%20(CC%20BY%204.0)-orange.svg)](#)
[![Test Accuracy: 88.74%](https://img.shields.io/badge/Test%20Accuracy-88.74%25-success.svg)](#)

> **ParkLens** is a transparent, explainable parking space occupancy detection system engineered for the university course *"Fundamentals of Computer Vision"*. Built strictly with **classical computer vision primitives** (homography warping, CLAHE equalization, Canny edge detection, and temporal background subtraction)—**no deep learning, no YOLO, and no black-box neural networks**.

---

## Key Capabilities & UI Preview

- **Live Overlay Annotation:** Green for Free, Red for Occupied, Yellow for error highlighting.
- **Top Dashboard Banner:** Real-time summary counts: `FREE: 71 | OCCUPIED: 29 | TOTAL: 100 (29% Full)`.
- **6-Stage Educational Pipeline Explorer:** Original $\to$ Grayscale $\to$ Gaussian Blur $\to$ CLAHE $\to$ Canny Edges $\to$ Difference Map $\to$ Decision.
- **Smart Parking Grid & Wayfinding:** Top-down slot tiles with nearest available space recommendations.
- **Diurnal Time-Lapse Timeline:** Interactive temporal scrubbing of lot occupancy across daily captures.
- **Performance Benchmark Dashboard:** Automated generation of Confusion Matrices, ROC Curves (AUC), and Failure Case Galleries.

---

## Project Architecture

```
ParkLens/
├── README.md # Main project overview and user documentation
├── requirements.txt # Python package dependencies
├── config.py # Centralized hyperparameters, thresholds & file paths
├── main.py # CLI orchestrator (download, prepare, tune, evaluate, predict)
├── app.py # Streamlit web presentation application
├── ParkLens.ipynb # Step-by-step educational Jupyter notebook for viva
├── .streamlit/
│ └── config.toml # Dark-theme styling (Teal accent, dark background)
├── scripts/
│ ├── download_data.py # High-speed multi-threaded Hugging Face data downloader
│ ├── make_synthetic.py # Synthetic parking lot generator (DEMO DATA fallback)
│ └── verify_data.py # Dataset integrity verification and sanity overlay script
├── src/
│ ├── acquisition.py # Dataset loading, XML contour parsing, 80/20 train/test split
│ ├── preprocessing.py # Homography warping, grayscale, Gaussian blur, CLAHE
│ ├── features.py # Canny edge density, background subtraction, Laplacian variance
│ ├── classifier.py # Rule-based decision logic, confidence scoring, grid-search tuning
│ ├── evaluation.py # Confusion matrix, accuracy, sensitivity, precision, F1, ROC/AUC
│ ├── visualization.py # Polygon overlays, pipeline composite, publication figures
│ └── utils.py # Coordinate ordering, polygon masking, seed initialization
├── ui/
│ ├── components.py # Reusable Streamlit HTML presentation components
│ ├── styles.css # Modern dark CSS stylesheet (cards, glow, rounded tiles)
│ └── charts.py # Interactive Plotly figures (donuts, feature bars, timelines)
├── data/
│ ├── PKLot/ # Acquired subset: PUCPR, UFPR04, UFPR05 across 3 weathers
│ ├── index.json # Normalized dataset index with exact polygon coordinates
│ ├── MANIFEST.json # Download provenance metadata, timestamps, and citations
│ ├── splits.json # Image-level 80/20 train/test split records
│ └── ref_patches.pkl # Per-stall canonical empty median background reference models
├── outputs/
│ ├── params.json # Tuned optimal hyperparameters per lot x weather cohort
│ ├── results.json # Test-set evaluation results JSON
│ ├── results.csv # Comprehensive cohort evaluation table CSV
│ ├── metrics.json # High-level summary metrics
│ ├── confusion_matrix.png# 2x2 confusion matrix heatmap figure
│ ├── roc_curves.png # Multi-lot ROC curves with AUC scores
│ └── weather_accuracy.png# Grouped bar chart comparing Sunny, Cloudy, Rainy
├── docs/
│ └── UI_GUIDE.md # 3-minute viva presentation demo script & screenshot checklist
└── report/
 ├── REPORT.md # Formal 8-page academic mini-project report
 └── VIVA_QA.md # 20 essential viva examination questions and answers
```

---

## Quick Start & Installation

### 1. Clone & Set Up Python Environment
Requires Python 3.10+.
```bash
cd ParkLens

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate # On Linux/macOS
.\.venv\Scripts\activate # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline (CLI)

The complete pipeline can be controlled through `main.py`:

```bash
# 1. Acquire PKLot dataset subset (450 images, 50 per lot x weather group)
python main.py download

# 2. Ingest annotations, perform 80/20 image-level split & build empty reference models
python main.py prepare

# 3. Grid-search hyperparameter tuning on train split (optimizes F1 per cohort)
python main.py tune

# 4. Evaluate quantitative metrics and generate plots on unseen test split
python main.py evaluate

# 5. Predict on a single image and export annotated visualization
python main.py predict --image data/PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_16_48_36.jpg --show-ids

# 6. Run synthetic fallback pipeline test (watermarked DEMO DATA)
python main.py demo

# 7. Execute complete end-to-end pipeline in one go
python main.py all
```

---

## Launching the Streamlit Demo Application

Launch the dark-themed presentation web app:
```bash
streamlit run app.py
```

### Interactive Tabs Overview:
1. ** Live Detection:** Real-time occupancy segmentation with instant KPI cards, donut chart, and CSV export.
2. ** Pipeline Explorer:** Step-by-step visual panels for any selected stall (Warp $\to$ Blur $\to$ CLAHE $\to$ Edges $\to$ Diff $\to$ Decision).
3. ** Parking Map:** Architectural slot grid view with automated nearest available spot routing.
4. ** Timeline:** Time-lapse scrubber tracking occupancy changes over diurnal sequence captures.
5. ** Performance Dashboard:** Test set metrics, confusion matrix, ROC curves, and failure case analysis.
6. **ℹ About:** Theoretical pipeline architecture diagram and academic references.

---

## Measured Test Set Results (Unseen Test Partition)

Evaluated strictly on **4,984 individual parking stall observations** across 90 images in the unseen 20% test split:

```
+-------------------------------------------------------------+
| PARKLENS BENCHMARK TEST METRICS |
+--------------------------+----------------------------------+
| Metric | Measured Score |
+--------------------------+----------------------------------+
| Overall Accuracy | 88.74% |
| Sensitivity (Recall) | 90.52% |
| Specificity | 87.04% |
| Precision | 86.99% |
| F1-Score | 88.72% |
| ROC Area Under Curve | 0.9374 |
+--------------------------+----------------------------------+
| True Positives (TP) | 2,216 |
| True Negatives (TN) | 2,206 |
| False Positives (FP) | 329 |
| False Negatives (FN) | 232 |
+--------------------------+----------------------------------+
| Total Stalls Tested | 4,984 |
+-------------------------------------------------------------+
```

---

## Data Acquisition Provenance

Data was autonomously downloaded via the official **Hugging Face Hub** mirror:
- **Repository:** `Voxel51/PKLot`
- **Schema:** Full $1280 \times 720$ camera images accompanied by exact polygonal contour coordinates per stall.
- **Downloaded:** 450 total images (50 per group across 3 lots $\times$ 3 weather conditions).
- **Storage:** Persisted locally to `./data/PKLot/<Lot>/<Weather>/<Date>/<name>.jpg` with corresponding `<name>.xml` files.

---

## Known Real-World Limitations

1. **Acute Solar Shadows:** Low-angle afternoon sunlight casts building eaves and light poles across empty stalls, producing false edge triggers (False Positives).
2. **Asphalt-Toned Paint:** Matte gray vehicles matching pavement luminance in overcast cloudy conditions may register lower edge gradients (False Negatives).
3. **Camera Vibration:** Perspective homography assumes rigid camera mounts; camera drift requires polygon re-calibration.

---

## Citations & Academic References

1. **PKLot Dataset:** 
 Almeida, P. R., Oliveira, L. S., Britto, A. S., Silva, E. J., & Koerich, A. L. (2015). 
 *PKLot – A robust dataset for parking lot classification.* Expert Systems with Applications, 42(11), 4937-4949. 
 DOI: [10.1016/j.eswa.2015.02.009](https://doi.org/10.1016/j.eswa.2015.02.009) (Licensed under CC BY 4.0).
2. **OpenCV (Open Source Computer Vision):** 
 Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal of Software Tools.
3. **VisionParkDetect Project Reference:** 
 Inspiration for classical edge and background difference pipeline formulation.
