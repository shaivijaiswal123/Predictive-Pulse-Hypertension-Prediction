"""
Test script to verify model works correctly
Run this to test the model without web interface
"""

import pickle
import numpy as np
import pandas as pd

print("="*60)
print("TESTING HYPERTENSION PREDICTION MODEL")
print("="*60)

# Load all required files
print("\n📁 Loading model files...")
model = pickle.load(open('logreg_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))
mappings = pickle.load(open('mappings.pkl', 'rb'))
print("✅ All files loaded successfully!")

# Test case 1: Normal Patient
print("\n" + "="*60)
print("TEST CASE 1: Normal Patient")
print("="*60)

test_patient_1 = {
    'Gender': 0,  # Male
    'Age_Group': 1,  # 18-34
    'Family_History': 0,  # No
    'Patient_Status': 0,  # No
    'Medication': 0,  # No
    'Symptom_Severity': 0,  # Mild
    'Shortness_Breath': 0,  # No
    'Visual_Changes': 0,  # No
    'Nosebleeds': 0,  # No
    'Time_Since_Diagnosis': 1,  # <1 Year
    'Systolic_BP': 115.5,  # 111-120 midpoint
    'Diastolic_BP': 85.5,  # 81-90 midpoint
    'Diet_Control': 1  # Yes
}

# Convert to array in correct order
features = []
for col in feature_names:
    features.append(test_patient_1[col])

# Scale and predict
features_array = np.array(features).reshape(1, -1)
features_scaled = scaler.transform(features_array)
prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

stage_names = ['Normal', 'Stage-1', 'Stage-2', 'Crisis']
print(f"\n📊 Prediction: {stage_names[prediction]}")
print(f"📊 Confidence: {max(probabilities)*100:.1f}%")
print("\n📊 Probability Distribution:")
for i, name in enumerate(stage_names):
    print(f"   {name}: {probabilities[i]*100:.1f}%")

# Test case 2: Stage-1 Patient
print("\n" + "="*60)
print("TEST CASE 2: Stage-1 Hypertension")
print("="*60)

test_patient_2 = {
    'Gender': 1,  # Female
    'Age_Group': 2,  # 35-50
    'Family_History': 1,  # Yes
    'Patient_Status': 1,  # Yes
    'Medication': 0,  # No
    'Symptom_Severity': 1,  # Moderate
    'Shortness_Breath': 1,  # Yes
    'Visual_Changes': 0,  # No
    'Nosebleeds': 0,  # No
    'Time_Since_Diagnosis': 2,  # 1-3 Years
    'Systolic_BP': 125.5,  # 121-130 midpoint
    'Diastolic_BP': 85.5,  # 81-90 midpoint
    'Diet_Control': 0  # No
}

features = []
for col in feature_names:
    features.append(test_patient_2[col])

features_array = np.array(features).reshape(1, -1)
features_scaled = scaler.transform(features_array)
prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

print(f"\n📊 Prediction: {stage_names[prediction]}")
print(f"📊 Confidence: {max(probabilities)*100:.1f}%")
print("\n📊 Probability Distribution:")
for i, name in enumerate(stage_names):
    print(f"   {name}: {probabilities[i]*100:.1f}%")

# Test case 3: Stage-2 Patient
print("\n" + "="*60)
print("TEST CASE 3: Stage-2 Hypertension")
print("="*60)

test_patient_3 = {
    'Gender': 0,  # Male
    'Age_Group': 3,  # 51-64
    'Family_History': 1,  # Yes
    'Patient_Status': 1,  # Yes
    'Medication': 1,  # Yes
    'Symptom_Severity': 2,  # Severe
    'Shortness_Breath': 1,  # Yes
    'Visual_Changes': 1,  # Yes
    'Nosebleeds': 0,  # No
    'Time_Since_Diagnosis': 3,  # 3-5 Years
    'Systolic_BP': 145.5,  # 141-150 midpoint
    'Diastolic_BP': 95.5,  # 91-100 midpoint
    'Diet_Control': 0  # No
}

features = []
for col in feature_names:
    features.append(test_patient_3[col])

features_array = np.array(features).reshape(1, -1)
features_scaled = scaler.transform(features_array)
prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

print(f"\n📊 Prediction: {stage_names[prediction]}")
print(f"📊 Confidence: {max(probabilities)*100:.1f}%")
print("\n📊 Probability Distribution:")
for i, name in enumerate(stage_names):
    print(f"   {name}: {probabilities[i]*100:.1f}%")

print("\n" + "="*60)
print("✅ MODEL TESTING COMPLETE!")
print("="*60)