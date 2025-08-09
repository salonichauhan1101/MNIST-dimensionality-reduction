# MNIST Dimensionality Reduction & Visualization

This project demonstrates three popular dimensionality reduction techniques — **PCA**, **t-SNE**, and a custom **Deep Autoencoder** — applied to the MNIST handwritten digits dataset. The goal is to compare their performance in creating 2D representations for visualization.

## Overview

We load and preprocess the MNIST dataset, apply each method, and visualize the results. This includes:

1. **PCA (Principal Component Analysis)** — Custom implementation from scratch.
2. **t-SNE (t-distributed Stochastic Neighbor Embedding)** — Using scikit-learn's `TSNE`.
3. **Deep Autoencoder** — Implemented from scratch with NumPy and trained using the Adam optimizer.

The results are compared based on:
- Clustering quality
- Representation strengths & weaknesses
- Computational performance

## Dataset

- **Source:** [MNIST dataset](https://www.openml.org/d/554)
- **Size:** 70,000 images, each 28x28 pixels, flattened to 784 features.
- **Labels:** Digits 0–9.

## Techniques

### 1. PCA (Custom)
- Implemented without scikit-learn's PCA class.
- Steps: Mean centering → Covariance matrix → Eigen decomposition → Projection.
- Strength: Fast, interpretable.
- Weakness: Limited to linear relationships.

### 2. t-SNE
- Applied to a subset of 10,000 samples (for efficiency).
- Captures local neighborhood structure well.
- Produces tight clusters but is computationally expensive.

### 3. Deep Autoencoder (Custom)
- Architecture:
    - Encoder: 784 → 128 → 64 → 2 (latent space)
    - Decoder: 2 → 64 → 128 → 784
- Trained using Adam optimizer from scratch in NumPy.
- Flexible and capable of learning non-linear representations.

## Visualizations
- **PCA:** Broad trends, overlapping clusters for similar digits.
- **t-SNE:** Clearer clusters, better local separation.
- **Autoencoder:** Learns non-linear embeddings, cluster quality depends on tuning.

## Requirements

```bash
numpy
matplotlib
scikit-learn
