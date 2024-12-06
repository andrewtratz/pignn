# CS229 Fall 2024 Final Project
## Physics-Informed Graph Neural Networks for Computational Fluid Dynamics

Author: Andrew Tratz (atratz@stanford.edu)

# Installation

* `pip install airfrans`

* Download AirfRANS dataset

* `pip install lips`

* `pip install lips.benchmark`

* `pip install -r requirements.txt`

* Clone ML4CFD starter kit from `https://github.com/IRT-SystemX/NeurIPS2024-ML4CFD-competition-Starting-Kit`

* Renamed starter kit folder to `Kit` for brevity

# Overview of files

* `config.py`: Global configuration constants - edit paths and hyperparameters here

* `data.py`: Collection of data manipulation helper functions

* `dataset.py`: AirfransGeo dataset used for preprocessing of AirfRANS data samples 

* `preprocess.py`: Script to preprocess train, cv, test, and ood_test datasets

* `loader.py`: Simple data loader class

* `interpolation.py`: Subroutines used for forward-pass interpolation of target variables over the mesh

* `physics.py`: Computation of partial derivatives (using pre-cached terms) and PINN loss functions

* `model.py`: Definition of graph neural network architecture

* `train.py`: Main training and cross-validation loop

* `inference.py`: Inference and evaluation script for test dataset

* `image.ipynb`: Jupyter notebook to produce graphics used in final report

