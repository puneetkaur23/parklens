# ParkLens: Project Technical Architecture and Teammate Guide

Welcome to **ParkLens**. This document is written for teammates joining the project. It explains exactly what code exists in this repository, how every module works, and how to present the project to instructors and examiners.

---

## 1. One-Minute Overview

### What Problem ParkLens Solves
Urban drivers spend an average of 10 to 15 minutes searching for parking, leading to congestion and carbon emissions. ParkLens automates parking space occupancy monitoring from existing static surveillance cameras mounted over outdoor parking facilities.

### Who Would Use It
- **Campus and Municipal Facility Managers:** To monitor real-time parking lot utilization without installing expensive in-ground magnetic or ultrasonic sensors ($300+ per space).
- **Smart-City Navigation Applications:** To route incoming drivers directly to vacant spaces via top-down guidance maps.

### What Goes In and What Comes Out
- **Input:** A single high-resolution image of a parking lot (such as a 1280x720 surveillance capture) and known 2D quadrilateral coordinates for each parking space boundary.
- **Output:** A binary classification for every space (0 = Free / Vacant, 1 = Occupied by a vehicle), an estimated confidence score (50% to 100%), and an annotated visual overlay (Green for Free, Red for Occupied, Yellow for errors).

### Why Classical Computer Vision (Not Deep Learning)
1. **Explainability:** In a university computer vision curriculum, understanding the fundamental physics and mathematics of 2D signals is critical. Deep neural networks (such as YOLO or Mask R-CNN) are black boxes that make predictions based on millions of uninterpretable weights. ParkLens relies entirely on explicit geometry, gradient operators, and intensity differences.
2. **Resource Efficiency:** ParkLens runs inference in real time on a standard CPU without requiring GPU acceleration or gigabytes of neural network weights.
3. **Deterministic Predictability:** Every decision is directly traceable to physical measurements: the density of sharp edges (car seams, grilles) and pixel deviation from empty pavement.

---

## 2. Project Map

Below is the directory tree with a one-line summary of each file and folder.

```
ParkLens/
|-- README.md                    System overview, project setup, and quick-start instructions.
|-- requirements.txt             Python package dependencies (OpenCV, NumPy, Streamlit, etc.).
|-- config.py                    Central hyperparameters, file paths, and default thresholds.
|-- main.py                      Command-line interface (download, prepare, tune, evaluate, predict).
|-- app.py                       Streamlit interactive web dashboard and demo application.
|-- ParkLens.ipynb               Step-by-step educational Jupyter notebook for coursework viva.
|-- .streamlit/
|   |-- config.toml              Streamlit theme configuration (dark background, teal accent).
|   `-- credentials.toml         Local headless credentials to suppress interactive CLI prompts.
|-- assets/
|   |-- favicon.png              Branded 64x64 browser favicon with the ParkLens "P" logo.
|   |-- favicon.svg              Vector SVG source of the branded favicon.
|   `-- backgrounds/             Offline local background images (bg1.jpeg, bg2.jpeg, bg3.jpeg).
|-- data/
|   |-- PKLot/                   Downloaded subset of the PKLot dataset (PUCPR, UFPR04, UFPR05).
|   |-- index.json               Standardized index of all 450 images and exact polygon coordinates.
|   |-- MANIFEST.json            Download metadata, timestamps, and academic citations.
|   |-- splits.json              Image-level 80/20 train/test split records (360 train, 90 test).
|   `-- ref_patches.pkl          Median empty background patches per lot space.
|-- outputs/
|   |-- params.json              Optimal tuned hyperparameters per lot and weather cohort.
|   |-- results.json             Test-set quantitative benchmark results JSON.
|   |-- results.csv              Complete breakdown table of test metrics across all 9 cohorts.
|   |-- metrics.json             High-level summary metrics (Accuracy, F1, ROC AUC).
|   |-- confusion_matrix.png     Annotated 2x2 test confusion matrix heatmap.
|   |-- roc_curves.png           Receiver Operating Characteristic curves with AUC scores.
|   `-- weather_accuracy.png     Grouped bar chart comparing performance across Sunny, Cloudy, Rainy.
|-- scripts/
|   |-- download_data.py         Multi-threaded Hugging Face downloader for PKLot subset.
|   |-- make_synthetic.py        Synthetic data generator used as offline fallback.
|   |-- verify_data.py           Dataset integrity verification and sanity overlay check.
|   `-- check_no_emoji.py        Automated scanner ensuring zero emojis across project files.
|-- src/
|   |-- __init__.py              Package initialization.
|   |-- acquisition.py           XML parsing, index creation, 80/20 splitting, reference synthesis.
|   |-- preprocessing.py         Perspective warping, grayscale, Gaussian blur, and CLAHE.
|   |-- features.py              Canny edge density and background subtraction discrepancy.
|   |-- classifier.py            Rule-based weighted linear scoring and parameter grid search.
|   |-- evaluation.py            Confusion matrix, precision, recall, F1-score, and ROC calculation.
|   |-- visualization.py         Polygon rendering, pipeline composites, and publication plots.
|   `-- utils.py                 Coordinate ordering, homography math, and file I/O helpers.
|-- ui/
|   |-- components.py            Reusable UI components (hero banner, KPI cards, parking grid).
|   |-- styles.css               Modern dark glass design system stylesheet.
|   `-- charts.py                High-contrast interactive Plotly visualizations (donuts, timelines).
`-- docs/
    |-- PROJECT_EXPLAINED.md     This comprehensive technical guide for teammates.
    |-- RESULTS_TABLE.md         Formal benchmark result tables formatted for project report.
    |-- RESULTS_TABLE.csv        CSV version of the evaluation results.
    |-- UI_PROGRESS.md           Design system progress checklist and design token audit.
    |-- UI_GUIDE.md              3-minute presentation script and screenshot checklist.
    `-- screenshots/             High-resolution report graphics at 1920x1080 and 1366x768.
```

### The 7 Core Files to Understand First
1. `config.py`: The single source of truth for all paths, default patch sizes, and tuning ranges.
2. `src/preprocessing.py`: How raw lot images are cropped and normalized against lighting shifts.
3. `src/features.py`: The mathematical core calculating edge density and background difference.
4. `src/classifier.py`: The rule-based decision formula and grid-search tuner.
5. `src/acquisition.py`: Dataset ingestion, image-level splitting, and empty reference patch synthesis.
6. `main.py`: The central command-line runner tying the backend pipeline together.
7. `app.py`: The Streamlit presentation app used during demonstrations.

---

## 3. The Pipeline, Step by Step

We walk through the pipeline using a real image from the dataset:
`data/PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_16_48_36.jpg` (1280x720 resolution, 100 parking spaces, 71 vacant and 29 occupied).

### Step 1: Ingestion & Geometric Rectification (Homography)
- **What it does:** Surveillance cameras view parking lots from high oblique angles, causing rectangular spaces to appear as skewed trapezoids (perspective distortion). This step extracts each 4-point space polygon and warps it into a standardized square patch ($64 \times 64$ pixels).
- **Exact function:** `warp_polygon_to_patch()` in `src/utils.py`.
- **Key parameters:** `PATCH_SIZE = (64, 64)`.
- **Implementation:** OpenCV calculates a $3 \times 3$ perspective transformation matrix $M$ using `cv2.getPerspectiveTransform()` and maps the source coordinates to $[(0,0), (64,0), (64,64), (0,64)]$ via `cv2.warpPerspective()`.

```python
# src/utils.py
rect = order_points(pts)
dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
M = cv2.getPerspectiveTransform(rect, dst)
return cv2.warpPerspective(image, M, (tw, th), flags=cv2.INTER_LINEAR)
```

### Step 2: Preprocessing (Grayscale, Blur, and CLAHE)
- **What it does:**
  1. *Grayscale:* Eliminates variable vehicle paint colors to focus strictly on luminance ($Y = 0.299R + 0.587G + 0.114B$).
  2. *Gaussian Blur:* Convolves the image with a $5 \times 5$ Gaussian kernel ($\sigma = 0$) to suppress high-frequency asphalt gravel noise.
  3. *CLAHE (Contrast Limited Adaptive Histogram Equalization):* Normalizes localized contrast. Global histogram equalization tends to over-amplify noise in dark shadows and wash out sunny pavement. CLAHE divides the patch into $8 \times 8$ contextual tiles, equalizes histograms locally, and clips histogram peaks at `clipLimit = 2.0` before redistributing pixels.
- **Exact function:** `preprocess_patch()` in `src/preprocessing.py`.
- **Key parameters:** `BLUR_KERNEL = 5`, `CLAHE_CLIP = 2.0`, `CLAHE_GRID = (8, 8)`.

```python
# src/preprocessing.py
def apply_clahe(gray: np.ndarray, clip_limit: float = CLAHE_CLIP, grid_size: Tuple[int, int] = CLAHE_GRID) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray)
```

### Step 3: Feature Extraction
ParkLens extracts two primary complementary features from each normalized $64 \times 64$ patch:

#### A. Canny Edge Density
- **Physical intuition:** Clean asphalt is textureless and exhibits few gradient changes. Vehicles consist of sharp geometric boundaries (hood seams, windshield frames, grilles, license plates, and tires) that generate dense edges.
- **Mathematical formula:**
  $$\text{Edge Density} = \frac{\text{Count of non-zero edge pixels in Canny binary map}}{\text{Total pixels in patch } (64 \times 64 = 4096)}$$
- **Exact function:** `edge_density()` in `src/features.py`.
- **Key parameters:** For PUCPR Sunny, `CANNY_LOW = 50`, `CANNY_HIGH = 180`.

```python
# src/features.py
edges = cv2.Canny(gray_patch, canny_low, canny_high)
total_pixels = gray_patch.shape[0] * gray_patch.shape[1]
edge_pixels = int(np.count_nonzero(edges))
ratio = float(edge_pixels) / float(total_pixels)
return float(np.clip(ratio, 0.0, 1.0)), edges
```

#### B. Background Subtraction Score
- **Physical intuition:** Stationary empty pavement resembles a precomputed empty reference model. An intruding car alters the intensity of large contiguous pixel blocks.
- **Mathematical formula:**
  $$\text{Diff Map } D(x, y) = |I_{\text{patch}}(x, y) - I_{\text{reference}}(x, y)|$$
  $$\text{Mean Diff} = \frac{1}{N} \sum \frac{D(x, y)}{255.0}, \quad \text{Excess Fraction} = \frac{\sum (D(x, y) > T_{\text{diff}})}{N}$$
  $$\text{Diff Score} = 0.5 \times \text{Mean Diff} + 0.5 \times \text{Excess Fraction}$$
- **Exact function:** `background_diff_score()` in `src/features.py`.
- **Key parameters:** For PUCPR Sunny, `DIFF_THRESHOLD = 45`.

```python
# src/features.py
diff_map = cv2.absdiff(patch_gray, ref_gray)
mean_diff = float(np.mean(diff_map)) / 255.0
excess_pixels = int(np.count_nonzero(diff_map > threshold))
excess_fraction = float(excess_pixels) / float(diff_map.size)
diff_score = 0.5 * mean_diff + 0.5 * excess_fraction
return float(np.clip(diff_score, 0.0, 1.0)), excess_fraction, diff_map
```

### Step 4: Decision Logic and Confidence Scoring
- **What it does:** Combines normalized edge density and background difference via equal weights (0.5 each) into a single scalar score $S \in [0.0, 1.0]$. If $S \ge T_{\text{decision}}$, the space is declared Occupied (1); otherwise Free (0).
- **Exact function:** `rule_based_classify()` in `src/classifier.py`.
- **Key parameters:** For PUCPR Sunny, `decision_threshold = 0.25`, `edge_weight = 0.5`, `diff_weight = 0.5`.
- **Confidence Formulation:**
  $$\text{Confidence} = 0.5 + 0.5 \times \frac{|S - T_{\text{decision}}|}{\max(T_{\text{decision}}, 1.0 - T_{\text{decision}})}$$
  This produces a score between 50% (uncertain border decision) and 100% (extreme separation).

```python
# src/classifier.py
score = float(w_e * ed + w_d * ds)
prediction = 1 if score >= decision_threshold else 0
denom = max(decision_threshold, 1.0 - decision_threshold, 1e-5)
margin = abs(score - decision_threshold)
confidence = float(np.clip(0.5 + 0.5 * (margin / denom), 0.5, 1.0))
return prediction, confidence, score
```

### Step 5: How the Background Reference Model is Built
- In `src/acquisition.py` (`build_reference_patches`):
  1. Filters the 80% training set for instances where ground truth is vacant (`occupied == 0`).
  2. Warps each vacant instance to $64 \times 64$ and converts to grayscale.
  3. Computes the **per-pixel median** across all vacant observations for that specific space ID:
     $$I_{\text{reference}}(x, y) = \text{median}(\{I_k(x, y) \mid k \in \text{Vacant Train Captures}\})$$
  4. Using the median instead of the mean prevents transient artifacts (such as a pedestrian walking across the spot in one training frame) from corrupting the reference pavement model.
  5. The resulting array of references is pickled to `data/ref_patches.pkl` (2.07 MB).

### Step 6: Dataset Split Strategy (Preventing Data Leakage)
- **Image-Level Partitioning:** `train_test_split_images()` in `src/acquisition.py` partitions by whole camera captures, never by individual parking spaces.
- **Why this is critical:** If spaces from the same camera frame were divided between train and test sets, the model would be tested on identical sun angles, shadows, and weather conditions, causing artificial spatiotemporal leakage.
- **Partition Ratio:** 80% train (360 images), 20% test (90 images), using fixed `RANDOM_SEED = 42`.

### Step 7: How Thresholds are Tuned
- `tune_all_groups()` in `src/classifier.py` runs a systematic grid search exclusively on the 80% training split:
  - Canny lower threshold: `[30, 40, 50, 70]`
  - Canny upper threshold: `[100, 120, 150, 180]`
  - Background diff threshold: `[20, 25, 30, 35, 45]`
  - Decision threshold: `[0.15, 0.20, 0.25, 0.30, 0.35, 0.40]`
- The optimizer maximizes the **F1-Score** on the training partition for each of the 9 lot x weather cohorts. The optimal parameters are saved to `outputs/params.json`.

### Step 8: Dataset Annotation Format and Limitations
- **Format:** PKLot XML annotation files contain exact quadrilateral contour points (`<contour><point x="..." y="..."/></contour>`).
- **Limitation:** Fixed 2D polygon masks assume camera mounts are rigid. If high winds or vibrations cause physical camera drift, the pre-calibrated space polygons drift out of alignment with the actual ground bays, requiring manual recalibration.

---

## 4. The User Interface Explained

The Streamlit web application (`app.py`) is styled with an offline dark-glass theme (`ui/styles.css`, `ui/components.py`).

### A. Header, Breadcrumb, and Status Strip
- **Hero Header:** Displays the ParkLens P-lens brand mark, wordmark, tagline, and a green `PKLot (Real Data)` badge.
- **Context Bar:** Monospace breadcrumb under the tabs tracking `Live Detection / PUCPR / Sunny / 2012-09-11_16_48_36.jpg`.
- **Status Strip:** Summarizes the active dataset partition, number of monitored spaces (e.g. 100), and active pipeline method.

### B. Sidebar Controls (Section by Section)
1. **Quick Demo Presets:** Pinned full-width buttons (`Sunny`, `Cloudy`, `Rainy`) for one-click demonstration.
2. **1. Data Ingestion:**
   - *Source Mode:* Toggle between "PKLot Dataset" (benchmark images) and "Upload Custom Image" (with optional XML).
   - *Parking Lot:* Select `PUCPR`, `UFPR04`, or `UFPR05`.
   - *Weather Condition:* Select `Sunny`, `Cloudy`, or `Rainy`.
   - *Timestamp:* Select specific chronological capture.
3. **2. Detection Method:** Select classifier architecture ("Rule-Based Weighted Linear").
4. **3. Model Tuning (Collapsible):**
   - *Reset to Tuned Values:* Restores the empirically tuned values from `outputs/params.json`.
   - *Canny Low / High:* Controls edge sensitivity. Increasing values ignores subtle edges (fewer false positives from asphalt texture, but risks missing cars). Decreasing values detects subtle seams (better car detection, but more shadow false alarms).
   - *Background Diff Threshold:* Minimum intensity change to mark a pixel changed. Increasing requires stronger contrast; decreasing makes it sensitive to faint lighting shifts.
   - *Decision Threshold:* Final cutoff for occupancy. Increasing demands higher combined edge/diff evidence (favors Specificity / fewer false alarms); decreasing favors Recall (catches every vehicle, but increases false alarms).
   - *Advanced Preprocessing:* Gaussian blur kernel (1 to 11) and CLAHE clip limit (1.0 to 5.0).
5. **4. Display & Diagnostics:** Toggles to show numeric Space IDs, show Confidence percentages, and highlight classification errors in yellow.
6. **Footer Controls:** Presentation Mode toggle (enlarges text and KPI numbers by ~12% for projectors) and "Reset All Settings" button.

### C. Tab 1: Live Detection
- **Display View Toggle:** Radio toggle between "Annotated Overlay" and "Raw Camera Capture" for instant before/after comparison.
- **Color Legend:** Green = Free Space, Red = Occupied Space, Yellow = Misclassified (shown when error highlighting is on).
- **KPI Cards:** Four responsive cards showing Total Spaces, Available Spaces, Occupied Spaces, and Capacity Utilization percentage.
- **Donut Chart:** Plotly interactive donut displaying the occupied share with total spaces centered in the hole.
- **Single-Frame Diagnostics:** Precision, Recall, and Accuracy against ground truth for that specific frame. If no occupied ground-truth spaces exist in that frame, Recall displays `N/A` with a tooltip explanation.
- **Exports:** Download buttons for the annotated image (PNG) and space data (CSV). The CSV contains: `Space_ID`, `Prediction`, `Confidence`, `Score`, `Ground_Truth`, `Correct`, `Edge_Density`, `Diff_Score`.
- **Space Details Table:** Collapsible searchable table beneath export buttons for inspecting individual stall scores.

### D. Tab 2: Pipeline Explorer
- **Filmstrip Component:** Horizontal cards with numbered badges (`STAGE 01` to `STAGE 07`) explaining the step-by-step image transformation.
- **Space Selector:** Dropdown to isolate any space ID in the image.
- **6-Step Visual Composite:** A horizontal strip rendering: 1. Warped BGR, 2. Grayscale, 3. Gaussian Blur, 4. CLAHE, 5. Canny Edges, 6. Difference Map, and 7. Final Decision Badge.
- **Space Inspector Deep Dive:** Three columns showing Observation vs. Empty Reference patch, a Plotly bar chart of extracted features (Edge Density, Diff Score, Laplacian Variance), and the binary decision logic.

### E. Tab 3: Parking Map
- **Bay Matrix:** Top-down architectural layout of rounded tiles. Free spaces pulse with a soft green glow; occupied spaces display vehicle outlines and certainty ratings.
- **Wayfinding:** Automatically sorts and recommends the nearest vacant spaces by vertical coordinate distance for arriving drivers.

### F. Tab 4: Timeline
- **Diurnal Fluctuation Curve:** Plotly interactive dual-line area chart tracking Available vs. Occupied spaces over chronological daily captures.
- **Scrubbing Slider:** Allows examiners to drag across timestamps and watch the lot fill and empty.

### G. Tab 5: Performance Dashboard
- **Metric Cards:** Summary of test split results (89.8% Accuracy, 88.6% F1, 90.5% Recall, 0.9510 ROC AUC).
- **Publication Figures:** 2x2 Confusion Matrix heatmap, multi-lot ROC curves, and grouped bar chart comparing Sunny, Cloudy, and Rainy accuracy.
- **Breakdown Table:** Full test metrics across all 9 cohorts.
- **Failure Cases Gallery:** Card grid showcasing false positives (sharp building shadows) and false negatives (asphalt-colored gray cars).

### H. Tab 6: About
- System methodology overview, academic citations for the PKLot dataset and OpenCV, and a Graphviz computational flowchart in dark theme.

---

### 3-Minute Live Viva Demo Script

Follow these 9 steps during a presentation:

1. **Minute 0:00 - Launch & Open:**
   Open the browser at `http://localhost:8501`. State: *"ParkLens is an explainable parking occupancy detector using classical computer vision—no deep learning black boxes."*
2. **Minute 0:25 - Preset Demonstration:**
   Click **Sunny** in the sidebar Quick Demo presets. Point to the annotated camera image: Green indicates free, Red indicates occupied.
3. **Minute 0:50 - Inspection Toggle:**
   Switch the display radio button from "Annotated Overlay" to "Raw Camera Capture" and back to demonstrate clean geometric overlay alignment.
4. **Minute 1:15 - Pipeline Explorer (Tab 2):**
   Click on the **Pipeline Explorer** tab. Walk through the 6-stage strip: *"Surveillance cameras view stalls obliquely, so we apply a 4-point perspective warp to standard 64x64 patches, normalize lighting via CLAHE, and extract Canny edges and background differences."*
5. **Minute 1:45 - Space Inspector:**
   Select Space `#1` and highlight the feature bars: *"An occupied space exhibits high edge density from vehicle seams and large pixel discrepancy against the empty pavement reference."*
6. **Minute 2:10 - Parking Map & Wayfinding (Tab 3):**
   Click on **Parking Map**. Show the pulsating green free bays and point to the "Nearest Free Spaces" recommendation list.
7. **Minute 2:30 - Timeline Scrubber (Tab 4):**
   Click on **Timeline**. Scrub the slider to show diurnal occupancy dynamics as cars arrive throughout the day.
8. **Minute 2:45 - Quantitative Evaluation (Tab 5):**
   Click on **Performance Dashboard**. State: *"Evaluated on 4,928 unseen test spaces across 90 images, ParkLens achieved 89.8% overall accuracy and 0.9510 ROC AUC."*
9. **Minute 2:55 - Failure Case Analysis:**
   Point to the failure cards: *"We transparently analyze failure modes: false alarms occur when low-angle solar shadows cast sharp edges across vacant pavement."* Conclude: *"Thank you. I am ready for questions."*

---

## 5. How to Run It (Commands from Fresh Clone)

```bash
# 1. Navigate to project root
cd "C:\Users\Puneet Kaur\.gemini\antigravity\scratch\ParkLens"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1   # On Windows PowerShell
# source .venv/bin/activate    # On Linux/macOS

# 3. Acquire PKLot dataset subset (50 images per group x 9 cohorts = 450 images)
python main.py download

# 4. Ingest annotations, create 80/20 image-level train/test split, and synthesize empty background models
python main.py prepare

# 5. Execute hyperparameter grid search on 80% training split
python main.py tune

# 6. Benchmark metrics and generate publication plots on 20% unseen test split
python main.py evaluate

# 7. Predict on a single image from CLI and export annotated JPEG
python main.py predict --image data/PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_16_48_36.jpg --show-ids

# 8. Launch presentation web app
streamlit run app.py
```

---

## 6. Known Limitations and Failure Cases

Honest analysis based on real test results:

1. **Acute Afternoon Sun Shadows (Primary False Positive Cause):**
   In PUCPR Sunny, low-angle sunlight casts high-contrast shadows of light poles and fences across vacant asphalt. Because Canny detects gradient transitions regardless of source, shadow edges cross the edge-density threshold, leading to 104 False Positives (Precision drops to 74.2%).
2. **Matte Gray Vehicle Camouflage (Primary False Negative Cause):**
   In overcast cloudy weather (notably UFPR04 Cloudy, where accuracy drops to 64.3% and precision to 20.0%), vehicles with matte gray or dark asphalt-toned paint have minimal luminance difference from pavement. When combined with flat diffuse lighting that eliminates shadows, edge and difference scores remain below the 0.25 threshold.
3. **Camera Angle and Vehicle Occlusion:**
   UFPR04 has a low, shallow camera angle. Taller SUVs in front rows occlude rear parking bays. In contrast, UFPR05 has a steep top-down perspective, achieving 94.4% accuracy and 0.9822 AUC.
4. **Rigid Camera Mount Assumption:**
   The homography matrix assumes the camera position never moves. If wind or mounting vibrations shift the camera, all 100 space polygons drift out of alignment.
5. **No Direct Inter-Vehicle Tracking:**
   ParkLens treats every frame as an independent observation; it does not currently maintain Kalman temporal tracking across continuous video streams.

---

## 7. Likely Teacher Questions and Short Answers

1. **Q: Why use classical computer vision instead of deep learning (YOLO, ResNet)?**
   *A:* Classical CV provides mathematical interpretability and full transparency. Every classification is traceable to physical geometric features (edges and pixel differences). It requires orders of magnitude less computation and runs on standard CPUs without GPU infrastructure.
2. **Q: Why did you split the dataset by IMAGE rather than by PARKING SPACE?**
   *A:* Splitting by individual spaces within the same image causes severe spatiotemporal data leakage because spaces in the same capture share identical atmospheric lighting, sun angles, and sensor noise. Partitioning at the image level ensures the test set evaluates unseen temporal captures.
3. **Q: Why use CLAHE instead of standard global histogram equalization?**
   *A:* Global histogram equalization shifts the whole image contrast, blowing out bright specular reflections on sunny days and amplifying noise in shadows. CLAHE operates on localized $8 \times 8$ tiles and clips histogram spikes at a contrast limit of 2.0 before redistributing pixels.
4. **Q: Why use the median instead of the mean to build the empty background reference?**
   *A:* The median is a robust statistical estimator with a 50% breakdown point. If a car or pedestrian temporarily intrudes on an empty space in a few training frames, the median completely ignores the outlier, whereas the mean would blend the vehicle into the reference pavement.
5. **Q: Why are thresholds tuned separately per parking lot and weather condition?**
   *A:* Different parking lots use different paving materials (dark asphalt at PUCPR vs. pale concrete pavers at UFPR05), and camera mounting distances vary. Weather conditions dramatically change dynamic range: Sunny captures require higher Canny thresholds (50/180) to ignore shadow borders, while Cloudy captures require lower thresholds (40/120) to detect low-contrast cars.
6. **Q: What is the difference between Sensitivity (Recall) and Specificity in this context?**
   *A:* Sensitivity (Recall) measures the fraction of actual occupied vehicles correctly detected ($\frac{TP}{TP + FN} = 90.5\%$). Specificity measures the ability to correctly recognize vacant spaces and not sound false alarms ($\frac{TN}{TN + FP} = 89.3\%$). For parking navigation, high Specificity is critical to avoid routing drivers to occupied spots.
7. **Q: What does the ROC Curve and AUC measure?**
   *A:* The Receiver Operating Characteristic (ROC) curve plots True Positive Rate against False Positive Rate across all continuous decision thresholds from 0.0 to 1.0. The Area Under Curve (AUC = 0.9510) measures the classifier's intrinsic ability to rank an occupied space higher than a vacant space regardless of chosen cutoff.
8. **Q: What mathematical operator does Canny use to compute image gradients?**
   *A:* Canny filters the image with first-derivative Sobel kernels ($K_x, K_y$) to compute gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$ and orientation $\theta = \arctan(G_y / G_x)$, followed by non-maximum suppression along gradient normals and hysteresis thresholding.
9. **Q: What is perspective homography and why is it needed?**
   *A:* Oblique surveillance perspectives distort rectangular bays into irregular quadrilaterals. A $3 \times 3$ homography matrix maps arbitrary quadrilateral coordinates in projective space $\mathbb{P}^2$ to a canonical affine $64 \times 64$ patch, normalizing physical scale across near and far stalls.
10. **Q: Why is UFPR04 the lowest-performing parking lot?**
    *A:* UFPR04 has the lowest camera elevation angle, causing significant inter-vehicle occlusion where cars in front rows block the view of stalls behind them, coupled with severe tree shadows.
11. **Q: How does ParkLens handle division-by-zero when displaying diagnostics?**
    *A:* In frames where no occupied ground-truth spaces exist ($TP + FN = 0$), Recall is mathematically undefined. The UI detects this zero denominator and displays `N/A` with a tooltip explanation rather than crashing or showing a misleading 0.0%.
12. **Q: What would be the immediate next step to improve accuracy?**
    *A:* Implementing a multi-frame temporal voting filter (requiring a stall to be classified occupied for 3 consecutive frames before changing state) and incorporating shadow-invariant color spaces (such as the invariant log-ratio of chromatic channels).

---

## 8. Who Explains What (Team Presentation Split)

### Teammate 1: Data Ingestion & Preprocessing
- PKLot dataset characteristics (PUCPR, UFPR04, UFPR05 across 3 weathers).
- Image-level 80/20 train/test split to prevent spatiotemporal leakage.
- 4-point perspective homography rectification to $64 \times 64$ canonical patches.
- Grayscale conversion, Gaussian noise filtering, and CLAHE illumination normalization.

### Teammate 2: Feature Extraction & Classification Logic
- Canny edge density formulation (why vehicles produce high edge counts vs. asphalt).
- Temporal median empty background reference synthesis and difference scoring.
- Weighted linear score combination ($S = 0.5 \cdot \text{Edge} + 0.5 \cdot \text{Diff}$).
- Per-cohort hyperparameter grid search tuning on the training set.

### Teammate 3: Quantitative Evaluation & Failure Modes
- Benchmark metrics on 4,928 test space observations (89.8% Accuracy, 0.9510 AUC).
- 2x2 Confusion Matrix interpretation (1,959 TP, 2,467 TN, 297 FP, 205 FN).
- Breakdown across weather conditions (why Cloudy performs best and Sunny has shadow false alarms).
- Detailed failure case analysis (acute afternoon solar shadows and gray car camouflage).

### Teammate 4: Interactive UI & Smart City Demo
- Live Detection view: overlay annotation, raw vs. annotated inspection toggle, and CSV export.
- Educational Pipeline Explorer: step-by-step filmstrip and Space Inspector.
- Parking Map architectural bay matrix and nearest available stall wayfinding.
- Timeline diurnal time-lapse scrubber showing daytime occupancy transitions.

