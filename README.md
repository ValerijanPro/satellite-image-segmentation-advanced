# Satellite Image Segmentation & Parcel Delineation (SAM-based)

An experimental **satellite image segmentation and agricultural parcel delineation pipeline**
combining modern foundation models (e.g. SAM) with classical computer vision and geospatial
post-processing techniques.

The project focuses on **real-world Earth Observation workflows**, where raw segmentation
outputs must be transformed into **geometrically meaningful parcel boundaries** suitable for GIS usage.

---

## Overview

Large segmentation models can produce accurate masks, but **their outputs are often not directly usable**
in geospatial applications.

This repository explores how segmentation masks can be refined using:
- edge detection
- line extraction
- geometric reasoning

and then converted into **vector formats** (Shapefiles) commonly used in GIS systems.

---

## Methodology & Results (Visual Overview)

The following animated presentation provides a **complete walkthrough of the approach**,
including motivation, processing steps, and qualitative results.

> ⚠️ Note: The presentation is in **Serbian**, but the visuals clearly illustrate the methodology.

![Pipeline Demo](demo.gif)

---

## Full Presentation (PDF)

For readers who prefer a **static, readable format**, the same content is available as a PDF:

📄 **[Download full presentation (PDF)](presentation.pdf)**

---

## Pipeline Summary

1. **Segmentation masks** obtained from a foundation model (e.g. SAM)
2. **Edge extraction** using Canny-based methods
3. **Line detection & structural refinement** via Hough Transform
4. **Parcel delineation** and geometric cleanup
5. **Export to GIS-compatible vector formats** (Shapefiles)

---

## Repository Contents

- `delineation.py`  
  Core parcel delineation logic from segmentation masks.

- `delineationCanny.py`  
  Edge-based delineation using Canny edge detection.

- `hough_transform.py`  
  Line detection and structural structure extraction using the Hough Transform.

- `tif_mask_to_shp.py`  
  Conversion of segmentation masks (`.tif`) into GIS-compatible Shapefiles.

- `tif_mask_to_shp_canny.py`  
  Alternative conversion pipeline with edge-based refinement.

---

## Data

The datasets used in this project are **not included** in the repository.

Typical inputs include:
- Sentinel-2 satellite imagery
- High-resolution segmentation masks produced by foundation models

Users interested in reproducing the pipeline should provide their own imagery
and adapt file paths accordingly.

---

## Context & Applications

This work is relevant to:
- agricultural parcel delineation
- land-use and land-cover analysis
- Earth Observation pipelines
- post-processing of foundation model outputs
- integration of deep learning with classical computer vision and GIS

---

## Notes

- This repository reflects **research and experimental work**, not a production-ready system.
- The focus is on **methodology and pipeline design**, rather than optimization.
- The project highlights the importance of **domain-aware post-processing**
  when applying large segmentation models to real-world geospatial data.
