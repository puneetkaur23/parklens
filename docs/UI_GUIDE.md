# ParkLens UI & Live Viva Presentation Guide

This guide provides a structured, presentation-tested **3-minute viva demonstration script**, what buttons to click in what order, what to say to the examiners, and a checklist of screenshots to include in your project report.

---

## 3-Minute Live Viva Presentation Script

### Minute 0:00 – 0:45 | Introduction & Live Detection (Tab 1)
1. **Launch the Application:**
 ```bash
 streamlit run app.py
 ```
2. **Opening Statement to Examiners:**
 > *"Good morning professors. Today I am presenting **ParkLens**, a parking space occupancy detector built using purely classical computer vision primitives. While modern deep learning models like YOLO act as black boxes, ParkLens demonstrates how fundamental geometric transformations, spatial edge analysis, and background subtraction can achieve robust, interpretable occupancy classification."*
3. **Action:**
 - Click the **Sunny** Quick Demo button on the sidebar.
 - Point to the live annotated image on the left.
4. **Talking Points:**
 - Explain the color code: **Green** means free, **Red** means occupied.
 - Point to the top banner: *"Notice the real-time tally: Free spots, Occupied spots, and Total occupancy rate."*
 - Point to the right-hand panel: *"On this current frame of 100 parking spaces, ParkLens achieves over 90% diagnostic accuracy. We can also toggle confidence ratings or download the per-space CSV results for downstream smart-city parking APIs."*

---

### Minute 0:45 – 1:45 | Educational Pipeline Journey (Tab 2)
1. **Action:**
 - Click on **Tab 2: Pipeline Explorer**.
 - Select Space `#1` or any occupied space from the dropdown.
2. **Talking Points:**
 - Walk the examiners through the 6-step horizontal composite:
 1. **Warp:** *"Because cameras observe parking lots from oblique elevated angles, rectangular spaces appear distorted. We apply a four-point perspective homography transform using OpenCV to warp each slot into a standardized 64×64 pixel patch."*
 2. **Grayscale & Gaussian Blur:** *"We suppress sensor noise using a 5×5 Gaussian filter."*
 3. **CLAHE:** *"Crucially, on sunny days or rainy overcast conditions, global illumination changes dramatically. We apply CLAHE—Contrast Limited Adaptive Histogram Equalization—which normalizes contrast in localized 8×8 tiles without blowing out noise."*
 4. **Canny Edges:** *"Look at the edge map: An occupied space contains sharp vehicle boundaries—hood seams, windshield contours, and tires—yielding high edge density. A vacant space has almost no edges except bare asphalt."*
 5. **Background Difference Map:** *"We subtract the current patch from our precomputed median 'empty reference' patch. The heatmap shows substantial pixel variance where the vehicle body sits."*
 6. **Decision:** *"Our linear combination of Edge Density and Background Difference cleanly crosses the tuned decision threshold."*

---

### Minute 1:45 – 2:30 | Smart City Parking Map & Time-Lapse (Tabs 3 & 4)
1. **Action:**
 - Click on **Tab 3: Parking Map**.
2. **Talking Points:**
 - *"Here we see how ParkLens translates raw pixel observations into a smart-parking dashboard. Every space is mapped to a colored tile. Drivers entering the parking facility are instantly routed to the nearest free spaces sorted by distance."*
3. **Action:**
 - Click on **Tab 4: Timeline**.
 - Drag the slider or scrub across timestamps.
4. **Talking Points:**
 - *"Here we track the temporal occupancy dynamics across a single day. As cars enter and leave between morning and afternoon, ParkLens tracks the filling of the lot seamlessly."*

---

### Minute 2:30 – 3:00 | Quantitative Validation & Limitations (Tab 5)
1. **Action:**
 - Click on **Tab 5: Performance Dashboard**.
2. **Talking Points:**
 - *"We rigorously evaluated ParkLens on an unseen 20% test split across 4,984 individual parking space observations from the PKLot dataset. It achieved an **Overall Accuracy of 88.74%**, an **F1-Score of 88.72%**, and an **ROC AUC of 0.9374**."*
 - Scroll down to the failure cases: *"We also transparently analyze failure cases: false positives typically occur when low-angle afternoon sunlight casts long building or tree shadows across vacant spots, creating edge artifacts. This provides clear rationale for future work, such as shadow-invariant color spaces or multi-frame temporal temporal voting."*
3. **Closing:**
 - *"Thank you, professors. I am now ready for your questions."*

---

## Report Screenshot Checklist

When compiling your final mini-project report, capture and embed the following screenshots from the UI:

| # | Screen / Widget | Location in UI | Purpose in Report |
|---|-----------------|----------------|-------------------|
| 1 | **Full Live Detection View** | Tab 1 (Main Screen) | Shows full-lot overlay with green/red spaces & KPI cards |
| 2 | **6-Step Pipeline Composite** | Tab 2 (Top Banner) | Visualizes the complete transformation from raw patch to decision |
| 3 | **Space Inspector Dual-View** | Tab 2 (Bottom) | Side-by-side comparison of warped observation vs. empty reference |
| 4 | **Smart Parking Grid** | Tab 3 | Illustrates top-down UI and nearest available spot recommendation |
| 5 | **Temporal Occupancy Line Chart**| Tab 4 | Shows diurnal curve of parking occupancy changes over time |
| 6 | **Confusion Matrix & ROC Curves**| Tab 5 | Empirical test-set validation figures |
| 7 | **Failure Case Gallery** | Tab 5 (Bottom) | Documents real-world challenges (shadows, reflections) |

