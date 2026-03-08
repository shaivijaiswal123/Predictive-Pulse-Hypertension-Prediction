"""
Predictive Pulse - Complete Training Pipeline
Phases 3-10: From Loading to Model Saving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
import pickle
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Create output directories
for phase in range(3, 11):
    os.makedirs(f'output/phase{phase}', exist_ok=True)

print("="*60)
print("PREDICTIVE PULSE - COMPLETE TRAINING PIPELINE")
print("="*60)

# ============================================================
# PHASE 3: DATASET LOADING
# ============================================================
print("\n" + "="*60)
print("📊 PHASE 3: DATASET LOADING")
print("="*60)

# Find dataset
data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
if not data_files:
    print("❌ No CSV file found in data/ folder!")
    exit()

dataset_path = os.path.join('data', data_files[0])
print(f"📁 Loading: {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Save raw data info
with open('output/phase3/raw_data_info.txt', 'w') as f:
    f.write(f"Dataset: {data_files[0]}\n")
    f.write(f"Rows: {df.shape[0]}\n")
    f.write(f"Columns: {df.shape[1]}\n")
    f.write(f"Columns: {', '.join(df.columns)}\n")

# ============================================================
# PHASE 4: DATA PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("🔄 PHASE 4: DATA PREPROCESSING")
print("="*60)

# Create a copy
df_clean = df.copy()

# Remove duplicates
duplicates = df_clean.duplicated().sum()
if duplicates > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"✅ Removed {duplicates} duplicates")

# Handle missing values
missing = df_clean.isnull().sum()
if missing.sum() > 0:
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    print("✅ Handled missing values")

# Strip whitespace from string columns
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = df_clean[col].str.strip()

print("\n📋 Original values before encoding:")
for col in df_clean.columns:
    print(f"\n{col}: {df_clean[col].unique()}")

# Create encoded dataframe
df_encoded = pd.DataFrame()

# 1. Encode Gender (C)
print("\n1️⃣ Encoding Gender (C):")
df_encoded['Gender'] = df_clean['C'].map({'Male': 0, 'Female': 1})
print(f"   Mapping: Male→0, Female→1")
print(f"   Values: {df_encoded['Gender'].unique()}")

# 2. Encode Age Group
print("\n2️⃣ Encoding Age Group:")
age_mapping = {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}
df_encoded['Age_Group'] = df_clean['Age'].map(age_mapping)
print(f"   Mapping: {age_mapping}")
print(f"   Values: {sorted(df_encoded['Age_Group'].unique())}")

# 3. Encode Binary Features
binary_features = {
    'History': 'Family_History',
    'Patient': 'Patient_Status',
    'TakeMedication': 'Medication',
    'BreathShortness': 'Shortness_Breath',
    'VisualChanges': 'Visual_Changes',
    'NoseBleeding': 'Nosebleeds',
    'ControlledDiet': 'Diet_Control'
}

print("\n3️⃣ Encoding Binary Features (Yes=1, No=0):")
for old_name, new_name in binary_features.items():
    df_encoded[new_name] = df_clean[old_name].map({'Yes': 1, 'No': 0})
    print(f"   {old_name} → {new_name}: {df_encoded[new_name].unique()}")

# 4. Encode Severity
print("\n4️⃣ Encoding Symptom Severity:")
severity_mapping = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
# Handle 'Sever' typo in data
df_clean['Severity'] = df_clean['Severity'].replace('Sever', 'Severe')
df_encoded['Symptom_Severity'] = df_clean['Severity'].map(severity_mapping)
print(f"   Mapping: {severity_mapping}")
print(f"   Values: {sorted(df_encoded['Symptom_Severity'].unique())}")

# 5. Encode Time Since Diagnosis
print("\n5️⃣ Encoding Time Since Diagnosis:")
time_mapping = {'<1 Year': 1, '1-3 Years': 2, '3-5 Years': 3, '5+ Years': 4}
# Handle variations in data
df_clean['Whendiagnoused'] = df_clean['Whendiagnoused'].replace('1 - 5 Years', '1-3 Years')
df_clean['Whendiagnoused'] = df_clean['Whendiagnoused'].replace('>5 Years', '5+ Years')
df_encoded['Time_Since_Diagnosis'] = df_clean['Whendiagnoused'].map(time_mapping)
print(f"   Mapping: {time_mapping}")
print(f"   Values: {sorted(df_encoded['Time_Since_Diagnosis'].unique())}")

# 6. Process Blood Pressure (convert ranges to midpoints)
print("\n6️⃣ Processing Blood Pressure:")

def bp_range_to_midpoint(bp_range):
    """Convert '111 - 120' to midpoint 115.5"""
    try:
        if pd.isna(bp_range):
            return np.nan
        # Clean the string
        bp_str = str(bp_range).replace(' ', '').replace('+', '')
        # Handle different formats
        if '-' in bp_str:
            parts = bp_str.split('-')
            if len(parts) == 2:
                low = float(parts[0])
                high = float(parts[1])
                return (low + high) / 2
        # Handle '100+' format
        elif bp_str.endswith('+'):
            return float(bp_str[:-1]) + 5
        else:
            return float(bp_str)
    except:
        return np.nan

df_encoded['Systolic_BP'] = df_clean['Systolic'].apply(bp_range_to_midpoint)
df_encoded['Diastolic_BP'] = df_clean['Diastolic'].apply(bp_range_to_midpoint)

print(f"   Systolic range: {df_encoded['Systolic_BP'].min():.1f} - {df_encoded['Systolic_BP'].max():.1f}")
print(f"   Diastolic range: {df_encoded['Diastolic_BP'].min():.1f} - {df_encoded['Diastolic_BP'].max():.1f}")

# 7. Encode Target (Stages)
print("\n7️⃣ Encoding Target Variable:")
stage_mapping = {
    'HYPERTENSION (Stage-1)': 1,
    'HYPERTENSION (Stage-2)': 2,
    'HYPERTENSIVE CRISIS': 3,
    'NORMAL': 0
}
# Handle variations in data
df_clean['Stages'] = df_clean['Stages'].replace('HYPERTENSION (Stage-2).', 'HYPERTENSION (Stage-2)')
df_clean['Stages'] = df_clean['Stages'].replace('HYPERTENSIVE CRISI', 'HYPERTENSIVE CRISIS')
df_encoded['Hypertension_Stage'] = df_clean['Stages'].map(stage_mapping)
print(f"   Mapping: {stage_mapping}")
print(f"   Values: {sorted(df_encoded['Hypertension_Stage'].unique())}")

# After encoding, check for and handle NaN values
print("\n🔍 Checking for NaN values after encoding:")
nan_counts = df_encoded.isnull().sum()
print(nan_counts[nan_counts > 0])

if nan_counts.sum() > 0:
    print("\n⚠️  NaN values detected! Handling them...")
    # Drop rows with NaN in target
    df_encoded = df_encoded.dropna(subset=['Hypertension_Stage'])
    
    # For feature columns, fill NaN with median
    for col in df_encoded.columns:
        if col != 'Hypertension_Stage' and df_encoded[col].isnull().any():
            df_encoded[col].fillna(df_encoded[col].median(), inplace=True)
    
    print(f"✅ NaN values handled. New shape: {df_encoded.shape}")

print("\n✅ Encoding Complete!")

# Save encoded data
df_encoded.to_csv('output/phase4/encoded_data.csv', index=False)
print("\n💾 Saved encoded data to output/phase4/encoded_data.csv")

# Save encoding mappings
mappings = {
    'age_mapping': age_mapping,
    'severity_mapping': severity_mapping,
    'time_mapping': time_mapping,
    'stage_mapping': stage_mapping,
    'binary_features': binary_features
}

with open('output/phase4/mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)
print("💾 Saved mappings to output/phase4/mappings.pkl")

# ============================================================
# FEATURE SELECTION AND SCALING
# ============================================================
print("\n" + "="*60)
print("🔧 FEATURE SELECTION AND SCALING")
print("="*60)

# Define features (exclude any non-feature columns)
feature_columns = [col for col in df_encoded.columns if col != 'Hypertension_Stage']
X = df_encoded[feature_columns]
y = df_encoded['Hypertension_Stage']

print(f"\n📋 Features ({len(feature_columns)}):")
for i, feat in enumerate(feature_columns, 1):
    print(f"   {i}. {feat}")

print(f"\n🎯 Target: Hypertension_Stage")
print(f"   Distribution:\n{y.value_counts().sort_index()}")

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

print("\n📏 Features scaled to [0, 1] range")

# Save scaler and feature names
with open('output/phase4/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('output/phase4/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("💾 Saved scaler and feature names to output/phase4/")

# ============================================================
# PHASE 5: EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "="*60)
print("📊 PHASE 5: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Gender Distribution
print("\n1️⃣ Creating Gender Distribution plot...")
plt.figure(figsize=(8, 6))
gender_counts = df_clean['C'].value_counts()
colors = ['#3498db', '#e74c3c']
plt.bar(gender_counts.index, gender_counts.values, color=colors, edgecolor='black')
plt.title('Gender Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Count')
for i, v in enumerate(gender_counts.values):
    plt.text(i, v + 5, str(v), ha='center')
plt.tight_layout()
plt.savefig('output/phase5/gender_distribution.png', dpi=300)
plt.close()
print("✅ Saved: gender_distribution.png")

# 2. Hypertension Stage Distribution
print("2️⃣ Creating Hypertension Stage Distribution plot...")
plt.figure(figsize=(10, 6))
stage_counts = df_clean['Stages'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
bars = plt.bar(range(len(stage_counts)), stage_counts.values, color=colors[:len(stage_counts)], edgecolor='black')
plt.title('Hypertension Stage Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Stage')
plt.ylabel('Count')
plt.xticks(range(len(stage_counts)), stage_counts.index, rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('output/phase5/stage_distribution.png', dpi=300)
plt.close()
print("✅ Saved: stage_distribution.png")

# 3. Correlation Heatmap
print("3️⃣ Creating Correlation Heatmap...")
plt.figure(figsize=(14, 10))
correlation = df_encoded.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/phase5/correlation_heatmap.png', dpi=300)
plt.close()
print("✅ Saved: correlation_heatmap.png")

# 4. Medication vs Severity
print("4️⃣ Creating Medication vs Severity plot...")
plt.figure(figsize=(10, 6))
med_sev = pd.crosstab(df_clean['TakeMedication'], df_clean['Severity'])
med_sev.plot(kind='bar', stacked=True, color=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title('Medication Usage vs Symptom Severity', fontsize=16, fontweight='bold')
plt.xlabel('On Medication')
plt.ylabel('Count')
plt.legend(title='Severity')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('output/phase5/medication_severity.png', dpi=300)
plt.close()
print("✅ Saved: medication_severity.png")

# 5. Age Group vs Hypertension Stage
print("5️⃣ Creating Age Group vs Hypertension Stage plot...")
plt.figure(figsize=(12, 7))
age_stage = pd.crosstab(df_clean['Age'], df_clean['Stages'])
age_stage.plot(kind='bar', stacked=True, color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'])
plt.title('Age Group vs Hypertension Stage', fontsize=16, fontweight='bold')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Stage')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('output/phase5/age_stage.png', dpi=300)
plt.close()
print("✅ Saved: age_stage.png")

# 6. Blood Pressure Analysis
print("6️⃣ Creating Blood Pressure Analysis plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Colors for stages
colors_map = {0: '#2ecc71', 1: '#f39c12', 2: '#e67e22', 3: '#e74c3c'}
stage_names = ['Normal', 'Stage-1', 'Stage-2', 'Crisis']

# Get unique stages and convert to integers
unique_stages = sorted(df_encoded['Hypertension_Stage'].unique())
print(f"   Unique stages found: {unique_stages}")

# Scatter plot
ax1 = axes[0, 0]
for stage_val in unique_stages:
    stage_int = int(stage_val)  # Convert to integer
    stage_data = df_encoded[df_encoded['Hypertension_Stage'] == stage_val]
    ax1.scatter(stage_data['Systolic_BP'], stage_data['Diastolic_BP'],
               c=colors_map[stage_int], label=stage_names[stage_int], alpha=0.6, s=30)
ax1.set_xlabel('Systolic BP (mmHg)', fontsize=11)
ax1.set_ylabel('Diastolic BP (mmHg)', fontsize=11)
ax1.set_title('BP Distribution by Stage', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Systolic distribution
ax2 = axes[0, 1]
for stage_val in unique_stages:
    stage_int = int(stage_val)
    stage_data = df_encoded[df_encoded['Hypertension_Stage'] == stage_val]
    ax2.hist(stage_data['Systolic_BP'], bins=20, alpha=0.5,
             color=colors_map[stage_int], label=stage_names[stage_int])
ax2.set_xlabel('Systolic BP (mmHg)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Systolic BP Distribution', fontsize=14, fontweight='bold')
ax2.legend()

# Diastolic distribution
ax3 = axes[1, 0]
for stage_val in unique_stages:
    stage_int = int(stage_val)
    stage_data = df_encoded[df_encoded['Hypertension_Stage'] == stage_val]
    ax3.hist(stage_data['Diastolic_BP'], bins=20, alpha=0.5,
             color=colors_map[stage_int], label=stage_names[stage_int])
ax3.set_xlabel('Diastolic BP (mmHg)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Diastolic BP Distribution', fontsize=14, fontweight='bold')
ax3.legend()

# Box plot
ax4 = axes[1, 1]
bp_data = []
bp_labels = []
for stage_val in unique_stages:
    stage_int = int(stage_val)
    bp_data.append(df_encoded[df_encoded['Hypertension_Stage'] == stage_val]['Systolic_BP'].dropna())
    bp_labels.append(stage_names[stage_int])

bp_box = ax4.boxplot(bp_data, labels=bp_labels, patch_artist=True)
for i, (patch, stage_val) in enumerate(zip(bp_box['boxes'], unique_stages)):
    stage_int = int(stage_val)
    patch.set_facecolor(colors_map[stage_int])
    patch.set_alpha(0.7)
ax4.set_xlabel('Hypertension Stage', fontsize=11)
ax4.set_ylabel('Systolic BP (mmHg)', fontsize=11)
ax4.set_title('Systolic BP by Stage', fontsize=14, fontweight='bold')

plt.suptitle('Blood Pressure Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('output/phase5/bp_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: bp_analysis.png")

# Save EDA summary
eda_summary = f"""
EDA SUMMARY
===========
Date: {datetime.now()}
Total Samples: {len(df_clean)}

Gender Distribution:
{df_clean['C'].value_counts()}

Stage Distribution:
{df_clean['Stages'].value_counts()}

Age Distribution:
{df_clean['Age'].value_counts().sort_index()}

Blood Pressure Ranges:
Systolic: {df_encoded['Systolic_BP'].min():.1f} - {df_encoded['Systolic_BP'].max():.1f}
Diastolic: {df_encoded['Diastolic_BP'].min():.1f} - {df_encoded['Diastolic_BP'].max():.1f}
"""

with open('output/phase5/eda_summary.txt', 'w') as f:
    f.write(eda_summary)

print("\n✅ All visualizations saved to output/phase5/")

# ============================================================
# PHASE 6: TRAIN-TEST SPLIT
# ============================================================
print("\n" + "="*60)
print("✂️ PHASE 6: TRAIN-TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Save split information
split_info = f"""
TRAIN-TEST SPLIT
================
Total samples: {len(X)}
Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)
Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)

Training set distribution:
{y_train.value_counts().sort_index()}

Testing set distribution:
{y_test.value_counts().sort_index()}
"""

with open('output/phase6/split_info.txt', 'w') as f:
    f.write(split_info)

# Save the splits
np.save('output/phase6/X_train.npy', X_train)
np.save('output/phase6/X_test.npy', X_test)
np.save('output/phase6/y_train.npy', y_train)
np.save('output/phase6/y_test.npy', y_test)
print("💾 Saved train-test splits to output/phase6/")

# ============================================================
# PHASE 7: MODEL TRAINING
# ============================================================
print("\n" + "="*60)
print("🤖 PHASE 7: MODEL TRAINING")
print("="*60)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Ridge Classifier': RidgeClassifier(random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Train models and save
trained_models = {}
training_results = []

for model_name, model in models.items():
    print(f"\n▶️ Training {model_name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Save model
    model_path = f'output/phase7/{model_name.replace(" ", "_").lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    trained_models[model_name] = model
    
    # Evaluate on training set
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    training_results.append({
        'Model': model_name,
        'Training_Accuracy': train_acc
    })
    
    print(f"   ✅ Training Accuracy: {train_acc:.4f}")
    print(f"   💾 Saved to: {model_path}")

# Save training results
train_results_df = pd.DataFrame(training_results)
train_results_df.to_csv('output/phase7/training_results.csv', index=False)
print("\n💾 Saved training results to output/phase7/training_results.csv")

# ============================================================
# PHASE 8: MODEL EVALUATION
# ============================================================
print("\n" + "="*60)
print("📈 PHASE 8: MODEL EVALUATION")
print("="*60)

evaluation_results = []
confusion_matrices = {}

for model_name, model in trained_models.items():
    print(f"\n▶️ Evaluating {model_name}...")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[model_name] = cm
    
    # Classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # Store results
    evaluation_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Save detailed evaluation
    with open(f'output/phase8/{model_name.replace(" ", "_").lower()}_report.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("="*40 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

# Create comparison dataframe
comparison_df = pd.DataFrame(evaluation_results)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False).round(4)
comparison_df.to_csv('output/phase8/model_comparison.csv', index=False)

print("\n📊 Model Comparison:")
print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.iloc[0]['Model']
best_accuracy = comparison_df.iloc[0]['Accuracy']
print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

# ============================================================
# PHASE 9: OVERFITTING ANALYSIS
# ============================================================
print("\n" + "="*60)
print("🔍 PHASE 9: OVERFITTING ANALYSIS")
print("="*60)

overfitting_results = []

for model_name, model in trained_models.items():
    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Test accuracy
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Gap
    gap = train_acc - test_acc
    
    overfitting_results.append({
        'Model': model_name,
        'Train_Accuracy': train_acc,
        'Test_Accuracy': test_acc,
        'Gap': gap,
        'Overfitting_Risk': 'HIGH' if gap > 0.10 or train_acc > 0.98 else 'LOW'
    })

overfitting_df = pd.DataFrame(overfitting_results)
overfitting_df.to_csv('output/phase9/overfitting_analysis.csv', index=False)

print("\n📊 Overfitting Analysis:")
print(overfitting_df.to_string(index=False))

print("\n📝 Analysis:")
print("""
- Models with >98% training accuracy may be overfitting
- Large gap (>10%) between train and test indicates overfitting
- Logistic Regression often generalizes better for medical data
- Simpler models are preferred for interpretability in healthcare
""")

with open('output/phase9/analysis_summary.txt', 'w') as f:
    f.write("OVERFITTING ANALYSIS\n")
    f.write("===================\n\n")
    f.write(overfitting_df.to_string())
    f.write("\n\n")
    f.write("""
In medical ML systems:
1. Overfitting can lead to misdiagnosis
2. Models should generalize to new patients
3. Interpretability is crucial for clinical acceptance
4. Logistic Regression balances performance and interpretability
    """)

# ============================================================
# PHASE 10: SAVE PRODUCTION MODEL
# ============================================================
print("\n" + "="*60)
print("💾 PHASE 10: SAVING PRODUCTION MODEL")
print("="*60)

# Save Logistic Regression as production model (as specified)
production_model = trained_models['Logistic Regression']

# Save to main directory for web app
with open('logreg_model.pkl', 'wb') as f:
    pickle.dump(production_model, f)

# Also save to output
with open('output/phase10/logreg_model.pkl', 'wb') as f:
    pickle.dump(production_model, f)

# Save preprocessing objects for web app
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

with open('mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)

print("✅ Production model saved:")
print("   - logreg_model.pkl (main folder)")
print("   - output/phase10/logreg_model.pkl")
print("✅ Preprocessing objects saved:")
print("   - scaler.pkl")
print("   - feature_names.pkl")
print("   - mappings.pkl")

# Test the saved model
print("\n🔍 Testing saved model...")
test_sample = X_test.iloc[0:1]
pred = production_model.predict(test_sample)[0]
proba = production_model.predict_proba(test_sample)[0]
print(f"   Test prediction: {stage_names[int(pred)]}")
print(f"   Confidence: {max(proba)*100:.1f}%")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("🎉 TRAINING PIPELINE COMPLETE!")
print("="*60)

final_summary = f"""
FINAL SUMMARY
=============
Date: {datetime.now()}
Dataset: {data_files[0]}
Original samples: {len(df)}
After cleaning: {len(df_clean)}
After encoding & NaN handling: {len(df_encoded)}
Features: {len(feature_columns)}
Models trained: {len(models)}
Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})
Production model: Logistic Regression

Output folders:
- output/phase3/: Dataset loading outputs
- output/phase4/: Preprocessing outputs
- output/phase5/: EDA visualizations
- output/phase6/: Train-test splits
- output/phase7/: Trained models
- output/phase8/: Evaluation results
- output/phase9/: Overfitting analysis
- output/phase10/: Production model
"""

print(final_summary)

with open('output/final_summary.txt', 'w') as f:
    f.write(final_summary)