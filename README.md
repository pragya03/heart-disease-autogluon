# Heart Disease Prediction using AutoGluon (AutoML)

![AutoML](https://img.shields.io/badge/AutoML-AutoGluon-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-red)

> A Machine Learning pipeline using [AutoGluon Tabular](https://auto.gluon.ai) to predict heart disease from patient data, with minimal manual effort.

---

## Overview

This project implements a fully automated machine learning pipeline to predict the presence of heart disease using tabular clinical data. The objective is to demonstrate the efficiency of AutoGluon AutoML for fast prototyping, high accuracy, and robust results.

---

## Dataset

- **Source**: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Samples**: 918 records
- **Target Variable**: `HeartDisease` (1 = presence, 0 = absence)

| Feature Name     | Description                      |
|------------------|----------------------------------|
| Age              | Age of patient                   |
| Sex              | Male/Female                      |
| ChestPainType    | Type of chest pain               |
| RestingBP        | Resting blood pressure           |
| Cholesterol      | Serum cholesterol (mg/dl)        |
| FastingBS        | Fasting blood sugar              |
| MaxHR            | Maximum heart rate               |
| ExerciseAngina   | Angina induced by exercise       |
| Oldpeak          | ST depression induced by exercise|
| ST_Slope         | Slope of peak exercise ST segment|

---

## Methodology

1. **Data Cleaning & Preprocessing**
2. **Train-Test Split** (80/20)
3. **Modeling with AutoGluon TabularPredictor**
4. **Baseline Comparison** (Random Forest)
5. **Evaluation Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
6. **Feature Importance** visualization

---

## Results

| Model             | Accuracy | F1 Score |
|-------------------|----------|----------|
| Random Forest     | 88%      | 0.86     |
| AutoGluon (default) | 85%      | 0.83     |
| AutoGluon (best_quality) | **89%** | **0.87** |

---

## Evaluation Metrics

- ✅ Accuracy
- 🔍 Precision / Recall
- 🎯 F1 Score
- 📉 Confusion Matrix

---

## Installation

```bash
git clone https://github.com/<your-username>/heart-disease-autogluon.git
cd heart-disease-autogluon
pip install -r requirements.txt
