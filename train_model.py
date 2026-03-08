"""
Predictive Pulse - Phase 3: Dataset Loading
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("="*60)
print("PREDICTIVE PULSE - PHASE 3: DATASET LOADING")
print("="*60)

# Create output directory for this phase
os.makedirs('output/phase3', exist_ok=True)

# STEP 1: Load the dataset
print("\n📊 STEP 1: Loading Dataset")
print("-"*40)

# Find dataset file in data folder
data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
if not data_files:
    print("❌ No CSV file found in data/ folder!")
    exit()

dataset_path = os.path.join('data', data_files[0])
print(f"📁 Loading: {dataset_path}")

try:
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# STEP 2: Display basic information
print("\n📋 Dataset Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

print("\n📋 First 5 rows:")
print(df.head())

# Save to file
df.head().to_csv('output/phase3/head.csv', index=False)
print("\n💾 Saved first 5 rows to output/phase3/head.csv")

# STEP 3: Dataset info
print("\n📋 Dataset Info:")
# Capture info as string
import io
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
print(info_str)

with open('output/phase3/info.txt', 'w') as f:
    f.write(info_str)

# STEP 4: Check missing values
print("\n🔍 Checking for missing values:")
missing = df.isnull().sum()
print(missing)

missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Values': missing.values,
    'Percentage': (missing.values / len(df) * 100).round(2)
})
missing_df.to_csv('output/phase3/missing_values.csv', index=False)
print("\n💾 Saved missing values report to output/phase3/missing_values.csv")

# STEP 5: Check for duplicates
print("\n🔍 Checking for duplicates:")
duplicates = df.duplicated().sum()
print(f"   Number of duplicate rows: {duplicates}")

with open('output/phase3/duplicates.txt', 'w') as f:
    f.write(f"Duplicate rows: {duplicates}\n")
    if duplicates > 0:
        f.write(f"Percentage: {(duplicates/len(df)*100):.2f}%")

# STEP 6: Basic statistics
print("\n📊 Basic Statistics:")
stats = df.describe(include='all').round(2)
print(stats)

stats.to_csv('output/phase3/statistics.csv')
print("\n💾 Saved statistics to output/phase3/statistics.csv")

# STEP 7: Data types
print("\n📊 Data Types:")
dtypes = pd.DataFrame({
    'Column': df.columns,
    'Data_Type': df.dtypes.values,
    'Unique_Values': [df[col].nunique() for col in df.columns]
})
print(dtypes)
dtypes.to_csv('output/phase3/data_types.csv', index=False)

# STEP 8: Value counts for categorical columns
print("\n📊 Value Counts for Categorical Columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
value_counts_dict = {}

for col in categorical_cols:
    print(f"\n{col}:")
    counts = df[col].value_counts()
    print(counts)
    value_counts_dict[col] = counts
    counts.to_csv(f'output/phase3/value_counts_{col}.csv')

# Summary report
summary = f"""
DATASET LOADING SUMMARY
=======================
Date: {datetime.now()}
Dataset: {data_files[0]}
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing Values: {missing.sum()}
Duplicates: {duplicates}
Categorical Columns: {len(categorical_cols)}
Numerical Columns: {len(df.select_dtypes(include=[np.number]).columns)}

Columns:
{', '.join(df.columns)}
"""

with open('output/phase3/summary.txt', 'w') as f:
    f.write(summary)

print("\n" + "="*60)
print("✅ PHASE 3 COMPLETE!")
print("📁 All outputs saved to output/phase3/")
print("="*60)