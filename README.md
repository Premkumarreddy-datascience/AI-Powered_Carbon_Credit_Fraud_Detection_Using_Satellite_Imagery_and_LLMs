<h1 align="center">AI-Powered Carbon Credit Fraud Detection Using Satellite Imagery and LLMs</h1>

<p align="center">
  <font color="gray" size="3">
    Detecting deforestation-based carbon credit fraud using satellite imagery, geospatial analysis, AI models, and intelligent risk scoring
  </font>
</p>

---

## Table of Contents

- [About the Project](#about-the-project)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Workflow Overview](#workflow-overview)
- [Data Sources](#data-sources)
- [Fraud Detection Methodology](#fraud-detection-methodology)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Installation Guide](#installation-guide)
- [Environment Variables](#environment-variables)
- [How to Run the Project](#how-to-run-the-project)
- [Outputs Generated](#outputs-generated)
- [Results and Interpretation](#results-and-interpretation)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

## About the Project

Carbon credit systems rely heavily on verified forest conservation and regeneration. However, fraudulent practices such as false afforestation claims, manipulated land boundaries, and unreported deforestation undermine the credibility of these systems.

This project presents an AI-powered framework that uses satellite imagery analysis, computer vision, anomaly detection, and risk scoring to identify potential carbon credit fraud.

The system is designed to be:
- Transparent
- Explainable
- Scalable

---

## Problem Statement

Carbon offset projects often claim forest preservation or regeneration, but:
- Satellite evidence is not always systematically analyzed
- Natural vs artificial deforestation patterns are hard to distinguish
- Boundary manipulation can falsely inflate carbon credits
- Manual audits are slow, expensive, and subjective

**There is a need for an automated, data-driven fraud detection system.**

---

## Project Objectives

This project analyzes multi-temporal satellite vegetation data to:
1. Measure forest health and cover using vegetation indices
2. Detect deforestation and abrupt changes
3. Identify suspicious spatial patterns
4. Quantify fraud risk using a weighted scoring mechanism
5. Generate interpretable visual dashboards and reports

Synthetic satellite data is used to demonstrate the full end-to-end pipeline in a reproducible and deployment-ready manner.

---

## Key Features

- Satellite-based land monitoring
- NDVI, EVI, NDWI vegetation analysis
- Before–After temporal change detection
- AI-based anomaly detection (Isolation Forest)
- Boundary & grid-pattern fraud detection
- Fraud risk scoring (0–1 scale)
- Rich visual dashboards & heatmaps
- End-to-end Jupyter Notebook pipeline

---

## Tech Stack

| Component            | Technology |
|---------------------|------------|
| **Programming Language**         | Python 3.10 |
| **Data Processing**  | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Geospatial**      | GeoPandas, Rasterio, Shapely |
| **Satellite APIs**| Sentinal Hub, Google Earth Engine |
| **Computer Vision**       | OpenCV, scikit-image |
| **Machine Learning**   | scikit-learn(Isolation Forest), PyTorch |
| **Notebooks**     | Jupyter Notebook |
| **Utilities**       | python-dotenv, tqdm |

---

## System Architecture
```
Satellite Data
     ↓
Vegetation Indices (NDVI, EVI, NDWI)
     ↓
Forest Segmentation & Metrics
     ↓
Change Detection (Before vs After)
     ↓
Fraud Pattern Analysis
     ↓
Risk Scoring Engine
     ↓
Visualization & Report Generation
```

---

## Workflow Overview

- Load satellite imagery (multi-temporal)
- Compute NDVI, EVI, NDWI
- Segment forest regions
- Compare forest cover over time
- Detect abrupt and spatial anomalies
- Identify boundary regularization & grid patterns
- Compute fraud risk score
- Generate dashboard & report

---

## Data Sources

- Sentinel-2 (simualted) for multispectral imagery
- Synthetic NDVI-based satellite data for demostration and reproducibility
- Configurable for real-world satellite APIs

> Synthetic data ensures the project runs **without API dependency**, while preserving real-world logic

---

## Fraud Detection Methodology

The fraud risk score is computed using multiple indicators:

| Indicator            | Description |
|---------------------|------------|
| **NDVI Change**         | Sudden vegetation loss |
| **Boundary Regularity**  | Artificial boundary smoothing |
| **Grid Pattern Score** | Planned clearing patterns |
| **Temporal Anaomalies**      | Deviations from seasonal norms |
| **Anamaly Detection**| Isolation Forest outliers |

---

## Notebook Walkthrough

The main notebook satellite_analysis.ipynb contains 17 structured cells, covering:

1. Setup and Configuration
2. Import Project modules
3. Intialize Modules
4. Define Area of Interest
5. Satellite Data generation
6. Create Synthetic Bands
7. Vegetation index calculation
8. Visualize Vegetation Indices
9. Forest Cover Analysis
10. Visualize Forest Cover
11. Change detection
12. Visualize Change detection
13. Fraud pattern detection
14. Fraud Risk Assessment
15. Generate Report
16. Final Visualization Dashboard
17. Saving Results in output folder

---

## Installation Guide

### Prerequisites

- Python 3.10+
- Anaconda (recommended)
- Git

### Setup

1. Clone the Repository
```bash
git clone<repo_url>
cd satellite_module
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Environment Variables
Create a **.env** file inside **satellite_module/**:

```env
SENTINELHUB_CLIENT_ID=YOUR_CLIENT_ID
SENTINELHUB_CLIENT_SECRET=YOUR_CLIENT_SECRET
EARTHENGINE_TOKEN=YOUR_EE_TOKEN
```

---

## How to Run the Project
Option 1: Run via Jupyter Notebook
```bash
jupyter notebook
```
Open:
```bash
notebooks/satellite_analysis.ipynb
```
Run cells sequentially from top to bottom

Option 2: Extend as a Python Pipeline
You can integrate the modules in **src/** into a production pipeline or API.

---

## Outputs Generated
After execution, the following files are saved:
```bash
data/outputs/
├── ndvi_before.npy
├── ndvi_after.npy
├── deforestation_mask.npy
└── analysis_dashboard.png
```

---

## Results and Interpretation
- **Forest Loss Detected**: Quantified in sq. km and %
- **Deforestation Pattern**: Natural (non-grid)
- **Boundary Changes**: Within normal range
- **Overall Fraud Risk**: LOW
- **Confidence Level**: ~87%
Project verified - no stron fraud indicators detected.

---

## Limitations
- Uses synthetic data for demonstration
- No real-time satellite streaming
- LLM integration currently conceptual
- Resolution limited to simulated 30m pixels

---

## Future Enhancements
- Real-time Sentinel-2 & Landsat ingestion
- LLM-based audit report generation
- Time-series forecasting of deforestation
- Web dashboard (FastAPI + React)
- Integration with carbon registries
- Blockchain-based verification logs

---

## Acknowledgements:

- European Space Agency (Sentinel-2)
- Google Earth Engine
- Open-source geospatial community
- scikit-learn, GeoPandas, OpenCV contributors
- Academic research on forest monitoring & carbon markets

---

## Author
- K. Prem Kumar Reddy
- M.Tech Data Science
- Jain (Deemed-to-be) University
