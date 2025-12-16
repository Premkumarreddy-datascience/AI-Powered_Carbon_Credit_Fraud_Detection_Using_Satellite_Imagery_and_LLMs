<h1 align="center">AI-Powered Carbon Credit Fraud Detection Using Satellite Imagery and LLMs</h1>

<p align="center">
  <font color="gray" size="3">
    Detecting deforestation-based carbon credit fraud using satellite imagery, geospatial analysis, AI models, and intelligent risk scoring
  </font>
</p>

---

## 📑 Table of Contents

- [About the Project](#about-the-project)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Workflow Overview](#workflow-overview)
- [Data Sources](#data-sources)
- [AI & ML Techniques Used](#ai--ml-techniques-used)
- [Fraud Detection Logic](#fraud-detection-logic)
- [Installation Guide](#installation-guide)
- [How to Run the Project](#how-to-run-the-project)
- [Jupyter Notebook Execution](#jupyter-notebook-execution)
- [Results & Outputs](#results--outputs)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)

---

## 🌿 About the Project

Carbon credit systems rely heavily on verified forest conservation and regeneration. However, fraudulent practices such as false afforestation claims, manipulated land boundaries, and unreported deforestation undermine the credibility of these systems.

This project presents an AI-powered framework that uses satellite imagery analysis, computer vision, anomaly detection, and risk scoring to identify potential carbon credit fraud.

The system is designed to be:
- Transparent
- Explainable
- Scalable

---

## ❗ Problem Statement

Carbon offset projects often claim forest preservation or regeneration, but:
- Satellite evidence is not always systematically analyzed
- Natural vs artificial deforestation patterns are hard to distinguish
- Boundary manipulation can falsely inflate carbon credits
- Manual audits are slow, expensive, and subjective

**There is a need for an automated, data-driven fraud detection system.**

---

## 🎯 Project Objectives

- Detect deforestation using satellite-derived vegetation indices
- Identify suspicious spatial and temporal patterns
- Quantify fraud risk using AI-based scoring
- Provide explainable, visual, and auditable results
- Enable scalable, repeatable monitoring without ground surveys

---

## ✨ Key Features

- 🌍 Satellite-based land monitoring
- 🌳 NDVI, EVI, NDWI vegetation analysis
- 🔄 Before–After temporal change detection
- 🧠 AI-based anomaly detection (Isolation Forest)
- 📐 Boundary & grid-pattern fraud detection
- 📊 Fraud risk scoring (0–1 scale)
- 📈 Rich visual dashboards & heatmaps
- 📓 End-to-end Jupyter Notebook pipeline

---

## 🛠️ Tech Stack

| Component            | Technology |
|---------------------|------------|
| **Language**         | Python 3.10 |
| **IDE / Platform**  | Jupyter Notebook |
| **Satellite Tools** | Sentinel Hub, Google Earth Engine |
| **Geospatial**      | GeoPandas, Rasterio, Shapely |
| **Image Processing**| OpenCV, scikit-image |
| **ML Models**       | Isolation Forest |
| **Visualization**   | Matplotlib, Seaborn, Plotly |
| **Environment**     | Conda / Virtualenv |
| **Standards**       | Reproducible research, modular design |

---

## 🧩 System Architecture

1. Area of Interest (AOI) definition  
2. Satellite data acquisition (real or synthetic)  
3. Vegetation index computation  
4. Forest cover segmentation  
5. Change & deforestation detection  
6. AI-based fraud pattern analysis  
7. Risk scoring & reporting  
8. Visualization & export  

---

## 🔄 Workflow Overview

- Load satellite imagery (multi-temporal)
- Compute NDVI, EVI, NDWI
- Segment forest regions
- Compare forest cover over time
- Detect abrupt and spatial anomalies
- Identify boundary regularization & grid patterns
- Compute fraud risk score
- Generate dashboard & report

---

## 🛰️ Data Sources

- Sentinel-2 Satellite Imagery
- Google Earth Engine (optional)
- Synthetic satellite data (for demonstration & reproducibility)

> Synthetic data ensures the project runs **without API dependency**, while preserving real-world logic.

---

## 🤖 AI & ML Techniques Used

| Technique | Purpose |
|---------|--------|
| NDVI / EVI / NDWI | Vegetation & water analysis |
| Change Detection | Forest loss identification |
| Isolation Forest | Unsupervised anomaly detection |
| Spatial Morphology | Noise & false-positive removal |
| Boundary Analysis | Fraud boundary manipulation detection |
| Pattern Detection | Grid-based deforestation detection |

---

## 🚨 Fraud Detection Logic

Fraud risk is calculated using multiple indicators:

- **Abrupt vegetation loss**
- **Boundary regularization**
- **Grid-like clearing patterns**
- **Spatial anomalies**
- **Temporal inconsistency**

### 🧮 Risk Score
- `0.0 – 0.3` → Low Risk  
- `0.3 – 0.7` → Medium Risk  
- `0.7 – 1.0` → High Risk  

Each factor is weighted and combined into a final score.

---

## ⚙️ Installation Guide

### ✅ Prerequisites

- Python 3.10+
- Anaconda (recommended)
- Git

### 🔧 Setup

```bash
# Clone the repository
git clone https://github.com/your-username/carbon-credit-fraud-detection
cd satellite_module

# Create environment
conda create -n carbonfraud python=3.10
conda activate carbonfraud

# Install dependencies
pip install -r requirements.txt
