# Flood-Detection-Sentinel-1
This repository features a flood detection project utilizing Sentinel-1 Synthetic Aperture Radar (SAR) images. The project aims to detect and analyze flood occurrences using SAR data for improved monitoring and early warning systems. The implementation includes a segmentation model for precise identification and delineation of flood-affected areas.

---

## Dataset Installation

Ensure to download all the necessary files to the `data` folder as indicated at the beginning of the `data_preparation.ipynb` file.

## Sentinel-1 Imagery

### Overview

This dataset comprises images collected by the Sentinel-1 Synthetic Aperture Radar (SAR) from various regions worldwide. The naming convention for Sentinel-1 images uses "vh" and "vv" to represent different polarizations of SAR data.

### Image Conversion

In the dataset, images are often converted to ratio and RGB formats for further analysis. A composite RGB image is generated using the following channels:

- Red: VV channel
- Green: VH channel
- Blue: VV/VH ratio

### Labeling

For model training purposes, additional images are provided for water body and flood labels. These labeled images are crucial for training machine learning models to accurately detect and analyze flood-related patterns.

### Visualizations

<p align="center">
<img src="https://github.com/Kacper0199/Flood-Detection-Sentinel-1/blob/main/images/sample_images.png" />
</p>

## Segmentation Model

A robust solution for flood prediction was developed by utilizing satellite images and employing the combination of a U-Net architecture with a ResNet encoder, implemented with the `segmentation-models-pytorch` library.

Installation:

```bash
pip install segmentation-models-pytorch
```

Create model:

```python
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=2, encoder_weights=None)
```

## Model Prediction Example

<p align="center">
<img src="https://github.com/Kacper0199/Flood-Detection-Sentinel-1/blob/main/images/sample_prediction.png" />
</p>
