# Cardiovascular Disease Risk Prediction

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

A machine learning course project for **CSE427: Machine Learning** that applies multiple classification and regression techniques to predict cardiovascular disease (CVD) risk levels from clinical and lifestyle features from a real-world dataset.

---

## Project Overview

This project trains and evaluates a range of machine learning models several implemented from scratch on a CVD clinical dataset. The task is a **three-class classification** problem (Low / Intermediary / High risk) with a parallel **regression task** predicting a continuous CVD Risk Score. A key clinical focus is minimizing false negatives in the High-risk class, where misclassification carries the greatest real-world consequence.

---

## Repository Structure

```
cardiovascular-disease-risk-prediction/
├── 11_12_24101406_23201572_22201334.ipynb   # Main analysis notebook (all models, EDA, evaluation)
├── CVD Dataset.csv                          # Training dataset (local copy)
├── Cardiovascular Disease Risk Prediction   # Full project report (LaTeX-compiled PDF)
│   [Report, Latex Format].pdf
└── README.md
```

---

## Dataset

**Cardiovascular Disease Risk Assessment Dataset**
[Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/cardiovascular-disease-risk-assessment-dataset/data) — DOI: [10.17632/d9scg7j8fp.1](https://data.mendeley.com/datasets/d9scg7j8fp/1) — License: CC BY 4.0

| Property | Value |
|---|---|
| Country / Region | Bangladesh |
| Data Source | Jamalpur Medical College Hospital |
| Collection Period | January 20, 2024 — January 1, 2025 |
| Total Records | 1,529 patient samples |
| Total Features | 22 clinical and lifestyle attributes |
| File Size | ~167 KB (CSV, UTF-8) |

The dataset captures real-world patient assessments from a tertiary hospital in Bangladesh, covering demographic, anthropometric, biochemical, and lifestyle risk factors relevant to South Asian cardiovascular epidemiology.

**Feature categories:** demographic (Sex, Age), anthropometric (Weight, Height, BMI, Abdominal Circumference, Waist-to-Height Ratio), clinical (Blood Pressure, Systolic BP, Diastolic BP), biochemical (Cholesterol, HDL, LDL, Fasting Blood Sugar, Estimated LDL), and lifestyle (Smoking, Diabetes, Physical Activity, Family History).

**Targets:**
- `CVD Risk Level` — categorical: `LOW`, `INTERMEDIARY`, `HIGH` (classification target)
- `CVD Risk Score` — continuous numeric (regression target)

**Preprocessing applied:**
- Systolic and diastolic BP recovered from composite `Blood Pressure (mmHg)` string column
- Height normalized from centimeters to meters where missing
- Invalid negative LDL values nullified
- BMI recalculated from weight and height where absent
- Numeric features: median imputation + standard scaling
- Categorical features: most-frequent imputation + one-hot encoding
- 80/20 stratified train/test split

---

## Technologies

| Category | Libraries / Frameworks |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib` |
| Classical ML | `scikit-learn` (Logistic Regression, KNN, Gradient Boosting, PCA, pipelines) |
| Deep learning | `TensorFlow` / `Keras` |
| Environment | Jupyter Notebook |

---

## Models Implemented

### Classification (CVD Risk Level)

| Model | Implementation | PCA Variant |
|---|---|---|
| Logistic Regression | scikit-learn | Yes |
| Random Forest | From scratch (bootstrap + decision stumps) | Yes |
| AdaBoost | From scratch (weighted stumps + alpha weighting) | Yes |
| Neural Network | TensorFlow/Keras (Dense 64→32→16→Softmax) | Yes |
| K-Nearest Neighbors | scikit-learn | After PCA only |
| Gradient Boosting | scikit-learn | After PCA only |

### Regression (CVD Risk Score)

| Model | Implementation |
|---|---|
| Random Forest Regressor | scikit-learn (200 estimators, max depth 8) |

---

## Key Features

- **From-scratch implementations** of Random Forest (bagging with bootstrap sampling) and AdaBoost (multi-class weighted stump ensemble), validated against their scikit-learn counterparts.
- **PCA dimensionality reduction** applied only to numeric features (retaining 95% variance), with categorical one-hot columns left untouched and recombined afterward.
- **Before vs. After PCA comparison** across all four shared models to quantify the effect of dimensionality reduction on accuracy.
- **False Negative (FN) analysis** targeting the HIGH-risk class — reports FN count, FN rate, and recall (sensitivity) per model, identifying the safest model for clinical screening use.
- **ROC AUC curves** plotted per class and macro-averaged for all 10 model variants (5 before PCA, 5 after PCA).
- **Confusion matrices** visualized separately for before-PCA and after-PCA model groups.
- **Regression residual and actual-vs-predicted plots** for the continuous CVD Risk Score.
- **Best model selection** determined by highest test accuracy across all model variants.

---

## Neural Network Architecture

```
Input → Dense(64, ReLU) → Dropout(0.2)
      → Dense(32, ReLU) → Dropout(0.2)
      → Dense(16, ReLU)
      → Dense(3, Softmax)

Optimizer : Adam (lr=0.001)
Loss      : Sparse Categorical Crossentropy
Training  : Up to 300 epochs, EarlyStopping (patience=15, restore best weights)
Batch size: 32
Split     : 70% train / 10% validation / 20% test
```

---

## Exploratory Data Analysis

The notebook covers:
- Descriptive statistics and data type audit
- Missing value and duplicate detection
- Histograms for all numerical features
- Bar charts for categorical features and target distribution
- Correlation heatmap for numerical features

---

## License

This project is licensed under the [MIT License](LICENSE).