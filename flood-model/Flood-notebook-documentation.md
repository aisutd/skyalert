# Flood Notebook: Task Documentation

## Purpose

`Flood.ipynb` is an initial data-loading and model-pipeline check for flood segmentation using the GEOID-Flood dataset. It downloads a small sample, displays satellite inputs alongside their labels, prepares the data, builds a U-Net, and checks whether the model can learn five selected image chips.

This is a development sanity check. It is not yet a validated flood-prediction system and does not measure performance on unseen areas or events.

## Dataset and files

The notebook uses the public [GEOID-Flood dataset](https://huggingface.co/datasets/links-ads/geoid-flood), specifically the sample folders for events `EMSR712-10` and `EMSR712-3`. It downloads sample Sentinel-1 GRD images, label masks, validity masks, and related files into a local `data/` directory.

The notebook inspects sample files and displays image/mask examples to help check that the inputs and labels can be opened and aligned. For the model setup and five-chip training check, it uses the `EMSR712-10` event.

## Data preparation

- Reads Sentinel-1 VV and VH radar bands from GeoTIFF files.
- Converts radar values to decibels and normalizes them using mean and standard deviation calculated from the training images.
- Uses label values for background, permanent water, and flood; pixels marked `255` are ignored by the training loss and IoU calculation.
- Chooses five 256 × 256 crops with the most flood-labeled pixels from the training tiles.

## Model and training check

The notebook creates a U-Net with a ResNet-34 encoder initialized with ImageNet weights. It accepts two input bands and produces three output classes per pixel. Cross-entropy loss ignores label value `255`; Adam updates the model weights.

For the Phase 5 check, the model trains for 200 rounds on the same five selected chips, then predicts those same chips. This deliberate small-sample overfit test checks whether the model and data pipeline can fit examples. It does not test generalization to new chips.

## Recorded Phase 5 results

The current saved notebook output reports:

| Tile | Flood-labeled pixels in selected crop |
|---|---:|
| EMSR712-10-12 | 51,141 |
| EMSR712-10-7 | 45,435 |
| EMSR712-10-13 | 29,951 |
| EMSR712-10-9 | 10,271 |
| EMSR712-10-15 | 5,081 |

- Input tensor shape: `[5, 2, 256, 256]`
- Label tensor shape: `[5, 256, 256]`
- Loss: `2.0296` at round 0; `0.0136` at the end of training
- Flood IoU on the same five training chips: `0.993`
- Tiny memorization check: **Pass**

The pass means the model closely reproduced the labels on the five chips it trained on. The reported IoU is not an independent evaluation score because it is calculated on those training chips.

## Current scope and next step

The notebook demonstrates loading and visualizing sample data and running an initial segmentation-model training check. It does not yet train and evaluate on separate training, validation, and test events. It also uses **after-flood Sentinel-1 imagery**, so this workflow maps observed flood extent; by itself, it does not forecast a flood before it begins.

A next modeling stage should split data by event or geographic area, train on the training split, tune/check on validation data, and report final metrics on held-out test data. A true advance warning workflow would also need pre-flood predictors and a clearly defined future prediction target.

## Running the notebook

Install the packages listed in `requirements.txt` from this folder, select that Python environment as the VS Code notebook kernel, then run the notebook cells in order. The first data-download cell requires internet access. The sample files are stored under `data/`.
