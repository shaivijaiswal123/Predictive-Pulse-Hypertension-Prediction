"""
Setup Verification Script
Run this to check if everything is installed correctly
"""

import sys
import os
from datetime import datetime

print("="*60)
print("PREDICTIVE PULSE - SETUP VERIFICATION")
print("="*60)

# Check Python version
print(f"\n📌 Python version: {sys.version}")

# Check if virtual environment is activated
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print(f"📌 Virtual Environment: {'✅ Active' if in_venv else '❌ Not active'}")

# Check installed packages
packages = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn', 'flask']
print("\n📌 Checking installed packages:")
for package in packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package}")

# Check folder structure
folders = ['data', 'output', 'static', 'templates']
print("\n📌 Checking folder structure:")
for folder in folders:
    if os.path.exists(folder):
        print(f"   ✅ {folder}/")
    else:
        print(f"   ❌ {folder}/ (creating...)")
        os.makedirs(folder, exist_ok=True)

# Check output subfolders
output_subfolders = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 
                     'phase6', 'phase7', 'phase8', 'phase9', 'phase10']
print("\n📌 Checking output subfolders:")
for subfolder in output_subfolders:
    path = os.path.join('output', subfolder)
    if os.path.exists(path):
        print(f"   ✅ output/{subfolder}/")
    else:
        print(f"   ❌ output/{subfolder}/ (creating...)")
        os.makedirs(path, exist_ok=True)

# Check if dataset exists
print("\n📌 Checking for dataset:")
dataset_files = [f for f in os.listdir('data') if f.endswith('.csv')] if os.path.exists('data') else []
if dataset_files:
    print(f"   ✅ Found dataset(s): {dataset_files}")
else:
    print(f"   ❌ No CSV files found in data/ folder. Please place your dataset there.")

# Save verification log
log_content = f"""
Setup Verification Log
Date: {datetime.now()}
Python Version: {sys.version}
Virtual Environment: {'Active' if in_venv else 'Not active'}
Packages: {[p for p in packages if __import__(p, globals(), locals(), [], 0)]}
Folders: {folders}
"""

with open('output/phase1/setup_verification.txt', 'w') as f:
    f.write(log_content)

print("\n✅ Verification complete! Log saved to output/phase1/setup_verification.txt")