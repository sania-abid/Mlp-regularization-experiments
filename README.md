# Mlp-regularization-experiments
MLP experiments: data augmentation, dropout, batch norm, k‑fold CV, L2 regularization.

# MLP Regularization and Optimization Experiments

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project explores several techniques to improve training and generalization of multi‑layer perceptrons (MLPs) on a synthetic binary classification dataset (`make_moons`). All experiments are implemented in **TensorFlow/Keras**.

The techniques investigated are:
- **Data Augmentation** (Gaussian noise + horizontal flips)
- **Dropout Regularization**
- **Batch Normalization** for faster convergence
- **Hyperparameter Tuning** with 5‑fold cross‑validation
- **L2 Regularization** (weight decay)

---

## Visual Results

### Original vs Augmented Data
<img width="554" height="197" alt="image" src="https://github.com/user-attachments/assets/059505f0-7cad-4700-b154-b9cc65a01152" />


### Dropout Experiment – Loss Curves
<img width="460" height="249" alt="image" src="https://github.com/user-attachments/assets/a6de7782-2270-49da-bb0e-caab14843ec1" />


---

## 🧪 Experiments Overview

### Data Augmentation
- **Goal:** Increase dataset diversity without collecting new data.
- **Method:** Gaussian noise (σ=0.15) and random horizontal flips (p=0.12) applied to `make_moons`.
- **Effect:** More varied samples help reduce overfitting.

### Dropout
- **Goal:** Prevent co‑adaptation of neurons.
- **Method:** 2‑hidden‑layer MLP (32,16) with dropout rate 0.5, compared to a model without dropout.
- **Result:** Dropout lowered validation loss and improved generalization.

### Batch Normalization
- **Goal:** Accelerate training and stabilize gradients.
- **Method:** Same MLP with BatchNormalization after each hidden layer.
- **Result:** BatchNorm converged faster (2 epochs vs. 3 to reach 90% accuracy) and smoothed loss.

### Hyperparameter Tuning with 5‑Fold CV
- **Goal:** Select the best hidden layer architecture.
- **Architectures:** (16,8), (32,16), (64,32) as hidden layers.
- **Result:** All performed similarly (~96.9% mean validation accuracy); the largest (64,32) was slightly best.

### L2 Regularization
- **Goal:** Penalize large weights to reduce overfitting.
- **Method:** MLP (64,32) trained with and without L2 penalty (λ=1e‑3).
- **Result:** L2 kept weight norms much smaller (6.56 vs. 27.0) and reduced the train‑val loss gap.

---

## 🚀 How to Run the Code

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
