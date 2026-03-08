"""
Diagnose why model gives same output for all inputs
"""

import pickle
import numpy as np
import pandas as pd
import os

print("="*60)
print("MODEL DIAGNOSTIC TOOL")
print("="*60)

# Load model and preprocessing objects
try:
    if os.path.exists('models/logreg_model.pkl'):
        model = pickle.load(open('models/logreg_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        feature_names = pickle.load(open('models/feature_names.pkl', 'rb'))
        print("✅ Loaded from models/ folder")
    else:
        model = pickle.load(open('logreg_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        print("✅ Loaded from root folder")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

print(f"\n📊 Model type: {type(model).__name__}")
print(f"📊 Features expected: {feature_names}")

# Test with different inputs
print("\n" + "="*60)
print("TESTING WITH DIFFERENT INPUTS")
print("="*60)

test_cases = [
    {
        'name': 'Case 1: Normal Patient',
        'data': {
            'Gender': 0, 'Age_Group': 1, 'Family_History': 0, 'Patient_Status': 0,
            'Medication': 0, 'Symptom_Severity': 0, 'Shortness_Breath': 0,
            'Visual_Changes': 0, 'Nosebleeds': 0, 'Time_Since_Diagnosis': 1,
            'Systolic_BP': 115, 'Diastolic_BP': 75, 'Diet_Control': 1
        }
    },
    {
        'name': 'Case 2: Stage-1 Patient',
        'data': {
            'Gender': 1, 'Age_Group': 2, 'Family_History': 1, 'Patient_Status': 1,
            'Medication': 0, 'Symptom_Severity': 1, 'Shortness_Breath': 1,
            'Visual_Changes': 0, 'Nosebleeds': 0, 'Time_Since_Diagnosis': 2,
            'Systolic_BP': 135, 'Diastolic_BP': 85, 'Diet_Control': 0
        }
    },
    {
        'name': 'Case 3: Stage-2 Patient',
        'data': {
            'Gender': 0, 'Age_Group': 3, 'Family_History': 1, 'Patient_Status': 1,
            'Medication': 1, 'Symptom_Severity': 2, 'Shortness_Breath': 1,
            'Visual_Changes': 1, 'Nosebleeds': 1, 'Time_Since_Diagnosis': 3,
            'Systolic_BP': 155, 'Diastolic_BP': 95, 'Diet_Control': 0
        }
    },
    {
        'name': 'Case 4: Crisis Patient',
        'data': {
            'Gender': 1, 'Age_Group': 4, 'Family_History': 1, 'Patient_Status': 1,
            'Medication': 1, 'Symptom_Severity': 2, 'Shortness_Breath': 1,
            'Visual_Changes': 1, 'Nosebleeds': 1, 'Time_Since_Diagnosis': 4,
            'Systolic_BP': 185, 'Diastolic_BP': 115, 'Diet_Control': 0
        }
    }
]

stage_names = ['Normal', 'Stage-1', 'Stage-2', 'Crisis']

for i, test in enumerate(test_cases):
    print(f"\n{'-'*40}")
    print(f"{test['name']}")
    print(f"{'-'*40}")
    
    # Create feature array in correct order
    features = []
    for col in feature_names:
        features.append(test['data'][col])
    
    features_array = np.array(features).reshape(1, -1)
    print(f"Features: {features_array}")
    
    # Scale
    features_scaled = scaler.transform(features_array)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    print(f"Prediction: {stage_names[prediction]} (Class {prediction})")
    print(f"Probabilities: Normal={probabilities[0]:.3f}, Stage-1={probabilities[1]:.3f}, Stage-2={probabilities[2]:.3f}, Crisis={probabilities[3]:.3f}")
    print(f"Confidence: {max(probabilities)*100:.1f}%")

# Check if all predictions are the same
print("\n" + "="*60)
print("CHECKING FOR MODEL ISSUES")
print("="*60)

# Check model coefficients (for Logistic Regression)
if hasattr(model, 'coef_'):
    print("\n📊 Model Coefficients:")
    coefs = model.coef_[0]
    for name, coef in zip(feature_names, coefs):
        print(f"   {name}: {coef:.4f}")
    
    # Check if all coefficients are near zero
    if np.all(np.abs(coefs) < 0.01):
        print("\n⚠️  WARNING: All coefficients are near zero! Model is not learning.")
else:
    print("\n📊 Model doesn't have coefficients (not Logistic Regression)")

# Check training data if available
if os.path.exists('output/phase6/y_train.npy'):
    y_train = np.load('output/phase6/y_train.npy')
    print(f"\n📊 Training data distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   Class {int(u)} ({stage_names[int(u)]}): {c} samples ({c/len(y_train)*100:.1f}%)")