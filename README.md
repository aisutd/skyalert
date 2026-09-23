# SkyAlert

## Description

SkyAlert is a computer vision system for detecting wildfires and floods from satellite imagery. It uses a classifier to flag affected regions, a segmentation model to map damage at pixel level, and engineered satellite features to provide actionable information through an interactive dashboard.

## Planned Technologies

- Python and JavaScript
- Streamlit or Gradio with Folium
- PyTorch, ResNet, U-Net, Rasterio, and scikit-learn
- NASA FIRMS, DeepGlobe, Google Earth Engine, and Copernicus EMS

## Docker Setup

Make sure Docker Desktop is installed and running.

```bash
docker compose up -d
docker compose down
docker compose ps
```

This is a generic starter configuration. The team can add project-specific dependencies and startup commands later.
