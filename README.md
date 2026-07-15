# Donut Mask Generation using Classical Computer Vision

## Overview

This project implements a classical computer vision pipeline for generating and refining segmentation masks of industrial donut objects without using any deep learning models.

The pipeline consists of two stages:

1. **Rough Mask Generation (Pre-processing)**
2. **Mask Refinement (Post-processing)**

The objective is to accurately segment donut-shaped objects while preserving their inner holes and minimizing background noise.

---

## Features

- Rough mask generation using HSV color thresholding
- ROI masking to remove unwanted regions
- Gaussian and Median filtering
- CLAHE-based illumination enhancement
- Morphological operations
- Connected component filtering
- Circularity-based contour filtering
- Donut hole preservation
- Optional shape completion for broken donuts
- Overlay visualization on original images

---

## Project Structure

```
.
├── images/                      # Input images
├── rough_masks/                 # Generated rough masks
├── refined_masks/               # Refined binary masks
├── completed_masks/             # Shape completed masks (optional)
├── overlays/                    # Refined mask overlays
├── overlays_completed/          # Completed mask overlays
├── visualizations/              # Comparison visualizations
├── results/                     # Generated reports/results
│
├── config.py                    # Configuration parameters
├── utils.py                     # Utility functions
├── generate_rough_masks.py      # Rough mask generation
├── refine_masks.py              # Mask refinement
├── visualize_results.py         # Visualization generation
│
├── requirements.txt
└── README.md
```

---

## Pipeline

### Stage 1 — Rough Mask Generation

The preprocessing stage performs:

- Region of Interest (ROI) masking
- Gaussian Blur
- Median Blur
- HSV color conversion
- CLAHE illumination enhancement
- HSV thresholding
- Morphological closing
- Connected component filtering
- Circularity filtering

Output:

- Binary rough masks

---

### Stage 2 — Mask Refinement

The postprocessing stage performs:

- Morphological cleanup
- Connected component filtering
- Circularity-based contour refinement
- Hole preservation
- Optional shape completion
- Overlay generation

Output:

- Refined masks
- Completed masks (optional)
- Overlay images

---

## Processing Workflow

```
Input Image
      │
      ▼
ROI Masking
      │
      ▼
Gaussian Blur
      │
      ▼
Median Blur
      │
      ▼
CLAHE
      │
      ▼
HSV Thresholding
      │
      ▼
Morphological Operations
      │
      ▼
Connected Components
      │
      ▼
Circularity Filtering
      │
      ▼
Rough Mask
      │
      ▼
Post Processing
      │
      ▼
Refined Mask
      │
      ▼
Optional Shape Completion
      │
      ▼
Overlay Generation
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Generate Rough Masks

```bash
python generate_rough_masks.py
```

### Refine Masks

```bash
python refine_masks.py
```

### Generate Visualizations

```bash
python visualize_results.py
```

---

## Outputs

The pipeline generates:

- Rough Masks
- Refined Masks
- Completed Masks (optional)
- Overlay Images
- Visualization Results

---

## Technologies Used

- Python
- OpenCV
- NumPy

---

## Notes

- This project uses only **classical computer vision techniques**.
- No deep learning or machine learning models are used.
- The pipeline is designed for industrial donut inspection datasets where preserving the donut hole is essential.

---