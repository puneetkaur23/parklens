# ParkLens Viva Voce Examination: 20 Essential Questions & Answers

### Course: *Fundamentals of Computer Vision*
**Project:** ParkLens – Classical Computer Vision Parking Occupancy Detector  
**Author:** Senior Computer Vision Student

---

### Q1: Why did you use classical computer vision instead of deep learning (e.g., YOLO or CNNs)?
**Answer:**  
In university computer vision education, the goal is to master the mathematical principles governing 2D image signals—geometry, gradient operators, frequency response, and statistical intensity distributions. Classical CV provides complete transparency and explainability: every decision is traceable to explicit physical features (edge density, pixel difference) rather than opaque neural network weights. Furthermore, classical algorithms require orders of magnitude less compute, zero GPU training infrastructure, and execute with deterministic predictability.

---

### Q2: Why did you split the dataset by IMAGE rather than by PARKING SPACE?
**Answer:**  
If we split randomly by individual parking spaces, spaces from the **same camera frame** would exist in both the training and testing sets. Because an image capture shares identical atmospheric lighting, solar angle, sensor noise, and weather conditions, evaluating on spaces from the same image causes severe **spatiotemporal data leakage**, artificially inflating test accuracy. Splitting strictly at the whole-image level guarantees that the test set evaluates unseen temporal captures under independent conditions.

---

### Q3: Why use CLAHE instead of standard Global Histogram Equalization?
**Answer:**  
Global Histogram Equalization flattens the cumulative intensity histogram across the entire image. In outdoor parking lots with direct sunlight and deep building shadows, global equalization over-amplifies background sensor noise in dark regions and blows out specular reflections on car metal.  
**CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) computes histograms over localized contextual tiles (e.g., $8 \times 8$). It applies a mathematical **clip limit** to the local histogram bins, redistributing clipped excess pixels uniformly before bilinear interpolation across tile boundaries, preventing noise over-amplification.

---

### Q4: Explain the 4 internal algorithmic stages of the Canny Edge Detector.
**Answer:**  
1. **Gaussian Smoothing:** Convolves the image with a Gaussian kernel to attenuate high-frequency sensor noise.
2. **Gradient Intensity & Direction:** Computes first spatial derivatives using Sobel kernels:
   $$G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan(G_y / G_x)$$
3. **Non-Maximum Suppression (NMS):** Thinning step where a pixel is preserved only if its gradient magnitude is a local maximum along the 1D line of its gradient direction $\theta$ (quantized to $0^\circ, 45^\circ, 90^\circ, 135^\circ$).
4. **Hysteresis Thresholding:** Uses dual thresholds ($T_{\text{low}}, T_{\text{high}}$). Pixels with $G > T_{\text{high}}$ are retained as strong edges; pixels with $G < T_{\text{low}}$ are suppressed; pixels between the two are retained only if they are 8-connected to a strong edge.

---

### Q5: How does the Perspective Homography Transform work mathematically?
**Answer:**  
A planar projective transformation (homography) maps coordinates between the perspective camera plane $(x, y)$ and the rectified canonical patch plane $(u, v)$ via a $3 \times 3$ matrix $\mathbf{H}$ with 8 degrees of freedom:
$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad u = \frac{x'}{w'}, \; v = \frac{y'}{w'}$$
Using 4 non-collinear corner points of the quadrilateral parking space polygon, we solve for $\mathbf{H}$ using `cv2.getPerspectiveTransform` and rectify the patch via `cv2.warpPerspective`.

---

### Q6: How do you construct the empty/reference background model without having an empty lot image?
**Answer:**  
Because complete empty-lot captures are rare in real operating parking facilities, we construct a **per-space temporal median model**. Across the training image set, for each individual parking stall ID, we collect all warped patches where ground truth is known to be FREE (empty). Taking the channel-wise median across time:
$$\mathbf{I}_{\text{ref}}(x, y) = \text{median}_{t} \left\{ \mathbf{I}_t(x, y) \right\}$$
The median filter robustly rejects transient objects (pedestrians, driving cars, seasonal debris) and isolates the static asphalt pavement texture.

---

### Q7: What is the physical intuition behind using Laplacian Variance?
**Answer:**  
The discrete Laplacian approximates the second spatial derivative:
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
Vacant asphalt pavement exhibits gradual intensity gradients, meaning second derivatives hover around zero with narrow dispersion. In contrast, an occupied vehicle introduces sharp structural transitions (chrome grilles, windshield wiper edges, tinted window margins), resulting in wide fluctuations in second derivatives and a significantly higher **variance** of the Laplacian.

---

### Q8: In a parking occupancy detector, what is the difference between Sensitivity and Specificity?
**Answer:**  
- **Sensitivity (Recall) = $\frac{\text{TP}}{\text{TP} + \text{FN}}$:** The fraction of parked cars correctly identified. High sensitivity prevents cars from being declared free when occupied (avoiding sending a driver to a taken spot).
- **Specificity = $\frac{\text{TN}}{\text{TN} + \text{FP}}$:** The fraction of vacant parking spots correctly identified as free. High specificity ensures available capacity is not under-reported.
In commercial smart-parking systems, high sensitivity is typically prioritized to prevent customer frustration.

---

### Q9: Why is parameter tuning performed separately per parking lot and per weather condition?
**Answer:**  
Different parking lots feature distinct asphalt textures, camera mount elevations, focal lengths, and viewing perspective angles. Different weather conditions dramatically change image statistics:
- **Sunny:** High dynamic range, strong shadow edges, bright specular car reflections.
- **Cloudy:** Diffuse illumination, muted contrast, reduced edge definitions.
- **Rainy:** Wet asphalt reflections, raindrops, droplet lens distortions.  
Tuning Canny thresholds and difference thresholds per cohort customizes the detector's sensitivity to the local signal-to-noise ratio.

---

### Q10: How does your weighted decision classifier calculate confidence?
**Answer:**  
The rule-based classifier computes a combined score $S = w_{\text{edge}} \cdot \text{EdgeDensity} + w_{\text{diff}} \cdot \text{DiffScore}$.  
Confidence represents the certainty of the binary declaration:
$$\text{Confidence} = 0.5 + 0.5 \times \frac{|S - \tau|}{\max(\tau, 1 - \tau)}$$
where $\tau$ is the decision threshold. If $S = \tau$, confidence is $50\%$ (maximum ambiguity). As $S$ diverges from $\tau$ towards $0.0$ or $1.0$, confidence approaches $100\%$.

---

### Q11: What primary factors cause False Positives (declaring an empty spot occupied)?
**Answer:**  
1. **Severe Shadows:** Acute building, light pole, or neighboring vehicle shadows cast sharp high-contrast boundary lines across bare asphalt, producing artificial Canny edges.
2. **Pavement Markings & Oil Stains:** Heavy painted arrows, handicap symbols, or dark oil leaks elevate background difference scores.
3. **Puddles:** On rainy days, standing water acts as a mirror, reflecting adjacent trees or buildings.

---

### Q12: What factors cause False Negatives (declaring an occupied spot free)?
**Answer:**  
1. **Asphalt-Colored Vehicles:** Matte gray, black, or dark silver cars with low-contrast paint matching asphalt luminance.
2. **Smooth Vehicle Profiles:** Modern aerodynamic car hoods lacking sharp trim, grilles, or seams inside the central warped patch.
3. **Overcast Low Contrast:** Diffuse lighting where edge gradients fail to exceed the Canny upper threshold $T_{\text{high}}$.

---

### Q13: What does the ROC Curve and AUC metric tell us about the detector?
**Answer:**  
The **Receiver Operating Characteristic (ROC)** plots True Positive Rate (Sensitivity) versus False Positive Rate ($1 - \text{Specificity}$) across all continuous classification thresholds from $0.0$ to $1.0$. The **Area Under Curve (AUC)** measures threshold-independent diagnostic discriminatory power:
- $\text{AUC} = 0.5$: Equivalent to random coin tossing.
- $\text{AUC} = 1.0$: Perfect classification separation.  
ParkLens achieves an empirical test AUC of **0.9374**, proving strong separability across classes.

---

### Q14: How does Gaussian blur kernel size affect the Canny edge detector?
**Answer:**  
The Gaussian kernel acts as a low-pass filter. Increasing kernel size ($3 \times 3 \to 5 \times 5 \to 7 \times 7$) suppresses higher frequencies:
- **Too small (e.g., $1 \times 1$):** High-frequency asphalt grain, camera sensor shot noise, and rain droplets generate spurious edges.
- **Too large (e.g., $11 \times 11$):** Subtle vehicle body contours and panel seams are blurred away, reducing true edge density.  
A $5 \times 5$ kernel was determined empirically to offer the optimal trade-off.

---

### Q15: Why is background subtraction alone insufficient for parking occupancy?
**Answer:**  
Background subtraction assumes a static background with stationary illumination. In outdoor environments:
- The sun moves continuously throughout the day, altering shadow geometry and ambient light temperature.
- Passing clouds cause rapid luminance fluctuations.
- A car painted the exact average gray value of the pavement might have low absolute intensity difference.  
Combining background difference with high-frequency spatial edge density creates a mutually complementary feature space.

---

### Q16: How are polygon vertices ordered prior to perspective transformation?
**Answer:**  
To compute a valid homography without self-intersecting or rotated projections, the four vertices must be sorted into canonical order: $[(x_{\text{TL}}, y_{\text{TL}}), (x_{\text{TR}}, y_{\text{TR}}), (x_{\text{BR}}, y_{\text{BR}}), (x_{\text{BL}}, y_{\text{BL}})]$.  
- Top-Left (TL) has the minimum sum $(x + y)$.
- Bottom-Right (BR) has the maximum sum $(x + y)$.
- Top-Right (TR) has the minimum difference $(y - x)$.
- Bottom-Left (BL) has the maximum difference $(y - x)$.

---

### Q17: What is the computational complexity of the ParkLens pipeline?
**Answer:**  
For an image with $N$ parking spaces:
- Homography warp: $\mathcal{O}(N \cdot W_p H_p)$ where $W_p \times H_p = 64 \times 64 = 4,096$ pixels.
- CLAHE: $\mathcal{O}(N \cdot W_p H_p)$.
- Canny edge detection: $\mathcal{O}(N \cdot W_p H_p)$ (linear in patch pixels).
- Absolute difference: $\mathcal{O}(N \cdot W_p H_p)$.  
Because patch size is fixed and small ($4,096$ pixels), total frame processing is $\mathcal{O}(N)$. On standard CPU hardware, ParkLens processes 100 stalls in less than $40\text{ ms}$ ($>25\text{ FPS}$), enabling real-time operation.

---

### Q18: What is the purpose of the synthetic dataset generator in scripts/make_synthetic.py?
**Answer:**  
In production and engineering pipelines, fallback generators provide unit testing coverage, sandbox isolation, and verification of GUI components before or in the absence of external network datasets. All synthetic data is watermarked with "DEMO DATA" to ensure academic integrity and ensure synthetic samples are never mixed into empirical research benchmark tables.

---

### Q19: How could you extend ParkLens with classical machine learning without deep learning?
**Answer:**  
Rather than applying heuristic linear weights $(w_{\text{edge}}, w_{\text{diff}})$, we can feed our 6-dimensional feature vectors:
$$\mathbf{x} = [\text{EdgeDensity}, \text{DiffScore}, \text{DiffExcess}, \text{LapVar}, \text{LuminanceStd}, \text{ColorDist}]^T$$
into a classical linear classifier such as **Logistic Regression** or a **Support Vector Machine (SVM)** with an RBF kernel. The SVM learns an optimal maximum-margin separating hyperplane in feature space, providing learned weighting while retaining computational speed.

---

### Q20: What are the main real-world deployment limitations of ParkLens?
**Answer:**  
1. **Fixed Camera Geometry:** Polygons are calibrated to a specific camera position. If the camera shifts due to wind or maintenance, re-calibration is required.
2. **Occlusion:** Vehicles in foreground rows can occlude parking stalls in background rows at low camera elevation angles.
3. **Severe Weather:** Snow covering pavement markings or ice glare obscuring lines alters baseline textures.
4. **Nighttime Operation:** Requires artificial lot lighting; infrared or low-light sensor tuning would be required for nocturnal operation.

