# Load and examine the real dataset provided by the user
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== LOADING REAL DATASET ===")

# Load the dataset
df = pd.read_csv('dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Dataset size: {len(df)} records")

# Examine the structure
print("\n=== DATASET STRUCTURE ===")
print("Columns:", df.columns.tolist())
print("\nColumn dtypes:")
print(df.dtypes)

# Display first few rows
print("\n=== FIRST 10 ROWS ===")
print(df.head(10))

# Basic statistics
print("\n=== BASIC STATISTICS ===")
print(df.describe())

# Check for missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Unique values in categorical columns
print("\n=== UNIQUE VALUES IN CATEGORICAL COLUMNS ===")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].nunique()} unique values")
    print(f"  Sample values: {df[col].unique()[:10]}")
    print()

# Check for date columns
print("\n=== DATE COLUMNS ANALYSIS ===")
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        print(f"Found potential date column: {col}")
        print(f"  Sample values: {df[col].head()}")
        print(f"  Data type: {df[col].dtype}")
        print()

print("=== INITIAL DATASET ANALYSIS COMPLETE ===")