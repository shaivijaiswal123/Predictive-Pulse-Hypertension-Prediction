"""
Train Model Script for Hypertension Prediction
Complete version with preprocessing, training, and evaluation
"""

# Import required libraries
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PREDICTIVE PULSE - HYPERTENSION PREDICTION SYSTEM")
print("="*60)

# STEP 1: Load the dataset
print("\n📊 STEP 1: Loading Dataset")
print("-"*40)

try:
    df = pd.read_csv('hypertension.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ Error: hypertension.csv not found!")
    exit()

# STEP 2: Remove duplicates
print("\n🔍 STEP 2: Removing Duplicates")
print("-"*40)

duplicates = df.duplicated().sum()
print(f"   Number of duplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   ✅ Removed {duplicates} duplicates")

print(f"   New shape: {df.shape}")

# STEP 3: Check missing values
print("\n🔍 STEP 3: Checking Missing Values")
print("-"*40)

missing = df.isnull().sum()
print(missing)

# STEP 4: Data Preprocessing
print("\n🔄 STEP 4: Data Preprocessing")
print("-"*40)

# Rename columns to ML-friendly names
df = df.rename(columns={
'C':'Gender',
'Age':'Age_Group',
'History':'Family_History',
'TakeMedication':'Medication',
'Severity':'Symptom_Severity',
'BreathShortness':'Shortness_Breath',
'VisualChanges':'Visual_Changes',
'NoseBleeding':'Nosebleeds',
'Whendiagnoused':'Time_Since_Diagnosis',
'Systolic':'Systolic_BP',
'Diastolic':'Diastolic_BP',
'ControlledDiet':'Diet_Control',
'Stages':'Hypertension_Stage'
})

print("   ✓ Columns renamed")

# Encode Gender
df['Gender'] = df['Gender'].map({
'Male':0,
'Female':1
})

# Encode Yes/No columns
binary_cols = [
'Family_History',
'Medication',
'Shortness_Breath',
'Visual_Changes',
'Nosebleeds',
'Diet_Control'
]

for col in binary_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].map({
    'Yes':1,
    'No':0
    })

# Encode Age
df['Age_Group'] = df['Age_Group'].map({
'18-34':1,
'35-50':2,
'51-64':3,
'65+':4
})

# Encode Severity
df['Symptom_Severity'] = df['Symptom_Severity'].map({
'Mild':0,
'Moderate':1,
'Severe':2,
'Sever':2
})

# Encode Time Since Diagnosis
df['Time_Since_Diagnosis'] = df['Time_Since_Diagnosis'].map({
'<1 Year':0,
'1-5 Years':1,
'5-10 Years':2,
'10+ Years':3
})

# Encode Target
df['Hypertension_Stage'] = df['Hypertension_Stage'].map({
'NORMAL':0,
'HYPERTENSION (Stage-1)':1,
'HYPERTENSION (Stage-2)':2,
'HYPERTENSIVE CRISIS':3
})

print("   ✓ Categorical features encoded")

# Convert BP ranges to numbers
def convert_bp(value):

    value = str(value)

    if "-" in value:
        low, high = value.split("-")
        return (float(low) + float(high)) / 2

    return float(value)

df['Systolic_BP'] = df['Systolic_BP'].apply(convert_bp)
df['Diastolic_BP'] = df['Diastolic_BP'].apply(convert_bp)

print("   ✓ Blood pressure converted to numeric")

print("\n   ✅ All features properly encoded!")

# STEP 5: Feature Selection
print("\n🔧 STEP 5: Feature Selection")
print("-"*40)

feature_columns = [
'Gender',
'Age_Group',
'Family_History',
'Medication',
'Time_Since_Diagnosis',
'Symptom_Severity',
'Shortness_Breath',
'Visual_Changes',
'Nosebleeds',
'Diet_Control',
'Systolic_BP',
'Diastolic_BP'
]

X = df[feature_columns]
y = df['Hypertension_Stage']

print(f"   Features ({len(feature_columns)}):")

for i,feat in enumerate(feature_columns,1):
    print(f"     {i}. {feat}")

print("\n   Target: Hypertension_Stage")

# STEP 6: Feature Scaling
print("\n📏 STEP 6: Feature Scaling (MinMaxScaler)")
print("-"*40)

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print("   ✅ Features scaled")

with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)

print("   💾 Scaler saved")

# STEP 7: Train Test Split
print("\n✂️ STEP 7: Train-Test Split")
print("-"*40)

X_train,X_test,y_train,y_test = train_test_split(
X_scaled,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print(f"   Training set size: {len(X_train)}")
print(f"   Testing set size: {len(X_test)}")

print("\n" + "="*60)
print("✅ PHASE 4 COMPLETE: Data Preprocessing Finished")
print("="*60)