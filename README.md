# BLISSNet for ERA5 Surface Temperature: Implementation and Cross-Region Evaluation

This repository contains an independent implementation of the BLISSNet architecture, applied to ERA5 2-metre temperature (t2m) reanalysis data over South Asia, Punjab, and Karnataka. It includes the model, training pipeline, dataset preparation notebook, and a set of inference experiments evaluating in-domain accuracy and cross-region generalization.

## Getting the data and model

Before running any notebook, obtain the dataset and, optionally, the trained model weights.

1. **Dataset (required).** Download the processed ERA5 NetCDF files from Kaggle and place them in `./datasets`:
   https://www.kaggle.com/datasets/thesyeddaniyal/era5-south-asia-t2m-6h-dataset-2025

2. **Model weights (optional).** If you do not want to run training yourself and only want to run inference, download the trained checkpoints from Kaggle and place them in `./models`:
   https://www.kaggle.com/models/thesyeddaniyal/blissnet-weights-for-era5-south-asia-t2m-dataset


## Repository structure

```
.
├── blissnet/                      Model implementation
│   └── blissnet.py                Trunk network, branch networks (Stage 1 and Stage 2), combined BLISSNet module
│
├── datasets/                      Processed ERA5 NetCDF files (downloaded, see above)
│
├── models/                        Trained checkpoints and training history (downloaded or produced by training.ipynb)
│
├── outputs/                       Inference outputs: reconstructed fields, metric records, visualizations (produced by inference.ipynb)
│
├── dataset_and_visualization.ipynb   Data preparation: GRIB to NetCDF, resampling, regional slicing, visualization
├── training.ipynb                    Two-stage training pipeline
├── inference.ipynb                   Cross-region inference and evaluation
└── README.md
```

## Background

BLISSNet is a DeepONet-style neural operator for reconstructing full spatial fields from sparse or gridded observations. It is trained in two stages. Stage 1 fits a SIREN trunk network, which produces a fixed set of basis functions over a coordinate domain, together with an Attention U-Net branch, which produces coefficients from a fully observed grid. Stage 2 freezes the trunk and the coefficient decoder from Stage 1, and trains a transformer encoder and fixed-size cross-attention block to produce the same coefficients from sparse, variable-count sensor observations. The output field is reconstructed as a weighted sum of the trunk's basis functions.

This implementation follows the architecture and training procedure described in the original BLISSNet paper (see Acknowledgments), adapted from the paper's synthetic fluid-flow setting (2D Navier-Stokes and Quasi-Geostrophic simulations on a fixed unit-square domain) to a geospatial setting using real ERA5 t2m reanalysis data over South Asia and two of its sub-regions.

## What was implemented and tested

- Full model architecture in PyTorch, including the Attention U-Net branch, SIREN trunk, transformer blocks with cross-attention, and the two-stage training loop with the four-term Stage 2 loss (control-point, coefficient, embedding, and ground-truth losses).
- A data pipeline from raw ERA5 GRIB files, via cfgrib, through resampling, regional slicing, and export to NetCDF, with interactive Plotly visualizations of the resulting fields.
- Three independently trained models, one per region (South Asia, Punjab, Karnataka), each trained for 15 epochs per stage.
- A 3x3 cross-region inference evaluation: each trained model was run against all three regions' test data, at 4x super-resolution, with RMSE, MAE, bias, and R² recorded for every combination.

## Architecture Sketch

```
Stage 1 (fully observed grid, per region)
------------------------------------------
  coordinates (x, y)                 t2m grid
        |                                |
        v                                v
  SIREN trunk network          Attention U-Net branch
  (fixed basis functions,             (coefficients)
   tied to region's                        |
   coordinate domain)                      |
        |                                  |
        +----------------+-----------------+
                          |
              reconstructed field =
        sum over basis functions, weighted
              by Stage 1 coefficients


Stage 2 (sparse, variable-count sensors)
------------------------------------------
  sparse sensor observations
        |
        v
  Transformer encoder -> fixed-size cross-attention
  (trunk and coefficient decoder frozen from Stage 1)
        |
        v
       coefficients
        |
        v
  reconstructed field = same trunk basis functions,
              weighted by Stage 2 coefficients
```

The trunk is what carries the region's coordinate domain. Stage 1 trains it jointly with the Attention U-Net branch on full grids. Stage 2 freezes the trunk and coefficient decoder, and trains a separate transformer + cross-attention encoder so the model can run from sparse, variable-count observations instead of a full grid. Both stages route through the same trunk basis functions, which is also why a trunk fit to one region's coordinates cannot be handed a different region's coordinates at inference and expected to work, as covered in Findings.

## Findings

**In-domain performance is strong.** Each model evaluated on its own training region achieves good reconstruction accuracy. For example, the South Asia model on South Asia reaches RMSE 2.33 K and R² 95.8%; the Punjab model on Punjab reaches RMSE 1.78 K and R² 82.3%. This matches the resolution-generalization behavior reported in the original paper, where a model trained at one grid resolution reconstructs accurately at coarser or finer resolutions over the same spatial domain.

**Cross-region generalization fails.** When a model trained on one region is evaluated on a different region it was never trained on, for example the Punjab-trained model evaluated on Karnataka, accuracy collapses: R² drops to 0% and RMSE and bias errors reach 20 to 35 K. This held consistently across all six cross-region combinations tested.

**Why this happens.** The original paper's zero-shot claim concerns resolution: a model trained on a fixed coordinate domain, such as a unit square, reconstructs accurately at grid resolutions it was never explicitly trained on, because the SIREN trunk is a continuous function of coordinates rather than a fixed-resolution grid. The paper does not claim, and does not test, generalization to a different spatial domain than the one the trunk was fit to. In this implementation, each of the three region-specific models has its own independently fitted trunk network, tied to that region's coordinate bounds. Evaluating a model outside its trunk's fitted coordinate domain is extrapolation for the SIREN trunk, a known weak point for coordinate-MLP architectures. This is compounded by the fact that the branch network, which has no explicit notion of location, also produces out-of-distribution coefficients when given a spatial input pattern unlike anything seen in training. As a result, both halves of the trunk-branch product are simultaneously unreliable, rather than showing the moderate degradation a partial mismatch might produce.

**Practical implication.** BLISSNet in this implementation is well suited to fast, repeated reconstruction over a fixed spatial domain from sparse or variable-count observations, including zero-shot super-resolution within that domain. It is not, without architectural changes such as a location-conditioned trunk, suited to a single model deployed across multiple disjoint regions it was not trained on.

## Requirements

- Python 3.10 or later
- PyTorch, NumPy, xarray, cfgrib, Plotly, Matplotlib, tqdm, kagglehub (for `download_assets.py`)

## Setup

Clone the repository and install dependencies from `requirements.txt`:

```
git clone https://github.com/syeddaniyalg/blissnet-era5-inference.git
cd blissnet-era5-inference
pip install -r requirements.txt
```

Then follow the "Getting the data and model" steps above before running any notebook.


## Acknowledgments

### Original paper

This implementation is based on the architecture and training methodology described in:

Veremchuk, M., Scott, K. A., and Pan, Z. BLISSNet: Deep Operator Learning for Fast and Accurate Flow Reconstruction from Sparse Sensor Measurements. arXiv:2602.24228, 2026.

The model design, two-stage training procedure, and loss formulation in `blissnet/blissnet.py` and `training.ipynb` follow this paper. All code in this repository, including the PyTorch implementation, the ERA5 data pipeline, the training runs, and the cross-region inference experiments, was independently written and run by the author of this repository (Syed Daniyal). The original paper's authors are not affiliated with this repository, and the results reported here, including the cross-region generalization findings above, have not been reviewed or endorsed by them.

### ERA5 data

This work uses ERA5 reanalysis data, produced by the Copernicus Climate Change Service (C3S) and obtained from the Copernicus Climate Data Store (CDS).

Generated using Copernicus Climate Change Service information (2025).

Contains modified Copernicus Climate Change Service information (2025). Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

Product citation:

Hersbach, H., Bell, B., Berrisford, P., et al. (2018): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.adbb2d47

ERA5 data is distributed under the Copernicus License (CC-BY), which permits free reuse, redistribution, and modification, subject to the attribution requirements above. See the Copernicus Climate Data Store licence for full terms: https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products

### This repository's dataset and model

The processed dataset and trained model weights hosted on Kaggle (linked above) are distributed under CC BY-SA 4.0, consistent with the attribution requirements of the underlying ERA5 data.