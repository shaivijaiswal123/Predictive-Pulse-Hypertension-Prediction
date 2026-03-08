# 🫀 Predictive Pulse - Hypertension Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Project Overview

Predictive Pulse is an advanced machine learning system that predicts hypertension stages using patient health data. The system implements and compares 7 different ML algorithms, with Logistic Regression selected as the production model for its interpretability and generalization capability.

### Hypertension Stages Classified:
- **Normal** 🟢 - Blood pressure within normal range
- **Stage-1** 🟡 - Early stage hypertension
- **Stage-2** 🟠 - Advanced hypertension
- **Crisis** 🔴 - Hypertensive crisis (emergency)

## ✨ Features

- **7 ML Models**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Ridge, Naive Bayes
- **Interactive Dashboard**: User-friendly medical web interface
- **Real-time Predictions**: Instant classification with confidence scores
- **Personalized Recommendations**: Stage-specific medical advice
- **Emergency Alerts**: Crisis detection with emergency protocols
- **Dark/Light Mode**: Theme toggle for user preference
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🏗️ Project Structure
PredictivePulse/
├── app.py # Flask web application
├── train_model.py # Complete ML pipeline
├── test_model.py # Model testing script
├── check_accuracy.py # Accuracy checker
├── requirements.txt # Python dependencies
├── README.md # Documentation
├── data/
│ └── hypertension_data.csv # Dataset
├── models/ # Trained models
│ ├── logreg_model.pkl
│ ├── scaler.pkl
│ └── feature_names.pkl
├── static/
│ └── style.css # Stylesheet
├── templates/
│ └── index.html # HTML template
├── output/ # All outputs
│ ├── phase3/ # Dataset loading
│ ├── phase4/ # Preprocessing
│ ├── phase5/ # EDA plots
│ ├── phase6/ # Train-test splits
│ ├── phase7/ # Trained models
│ ├── phase8/ # Evaluations
│ ├── phase9/ # Overfitting analysis
│ └── phase10/ # Production model
└── screenshots/ # UI screenshots


## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/shaivijaiswal123/Predictive-Pulse-Hypertension-Prediction.git
cd Predictive-Pulse-Hypertension-Prediction

📊 Model Performance
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	92.4%	92.1%	92.4%	92.2%
Random Forest	94.2%	94.0%	94.2%	94.1%
Decision Tree	88.7%	88.5%	88.7%	88.6%
SVM	91.3%	91.1%	91.3%	91.2%
KNN	87.5%	87.3%	87.5%	87.4%
Ridge	90.1%	89.9%	90.1%	90.0%
Naive Bayes	84.6%	84.4%	84.6%	84.5%
Production Model: Logistic Regression (selected for interpretability)

💻 Usage Guide
Web Interface
Fill patient information in the form

Click "Analyze Patient Data"

View results with color-coded risk levels

Follow medical recommendations