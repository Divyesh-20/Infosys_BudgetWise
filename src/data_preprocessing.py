# Data Cleaning and Preprocessing for the Real Dataset
import pandas as pd
import numpy as np
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

print("=== PHASE 1: DATA CLEANING & PREPROCESSING ===")

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Clean and standardize the amount column
print("1. Cleaning Amount Column...")
def clean_amount(amount):
    """Clean amount column by removing currency symbols and converting to float"""
    if pd.isna(amount):
        return np.nan
    
    # Convert to string and remove currency symbols
    amount_str = str(amount)
    amount_str = re.sub(r'[₹Rs\.\,]', '', amount_str)
    
    try:
        return float(amount_str)
    except:
        return np.nan

df['amount_cleaned'] = df['amount'].apply(clean_amount)

# Remove rows with invalid amounts
before_cleaning = len(df)
df = df.dropna(subset=['amount_cleaned'])
after_cleaning = len(df)
print(f"Removed {before_cleaning - after_cleaning} rows with invalid amounts")

# 2. Clean and standardize date column
print("\n2. Cleaning Date Column...")
def parse_date(date_str):
    """Parse various date formats"""
    if pd.isna(date_str):
        return np.nan
    
    date_str = str(date_str).strip()
    
    # Try different date formats
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y', 
        '%d/%m/%Y',
        '%d-%m-%y',
        '%d-%m-%Y',
        '%Y/%m/%d',
        '%m-%d-%Y'
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    # Try pandas automatic parsing
    try:
        return pd.to_datetime(date_str)
    except:
        return np.nan

df['date_cleaned'] = df['date'].apply(parse_date)

# Remove rows with invalid dates
before_date_cleaning = len(df)
df = df.dropna(subset=['date_cleaned'])
after_date_cleaning = len(df)
print(f"Removed {before_date_cleaning - after_date_cleaning} rows with invalid dates")

# 3. Standardize categories to match our project categories
print("\n3. Standardizing Categories...")

# Map various category spellings to standard categories
category_mapping = {
    # Education
    'education': 'education',
    'educaton': 'education',
    'Education': 'education',
    'EDUCATION': 'education',
    
    # Entertainment  
    'entertainment': 'entertainment',
    'Entertainment': 'entertainment',
    'ENTERTAINMENT': 'entertainment',
    
    # Food
    'food': 'food',
    'foods': 'food',
    'Food': 'food',
    'FOOD': 'food',
    'Fod': 'food',
    'Foods': 'food',
    'FOOd': 'food',
    
    # Health
    'health': 'health',
    'Health': 'health',
    'HEALTH': 'health',
    'healthcare': 'health',
    'medical': 'health',
    
    # Investment
    'investment': 'investment',
    'Investment': 'investment',
    'INVESTMENT': 'investment',
    'investments': 'investment',
    
    # Others
    'others': 'others',
    'Others': 'others',
    'OTHER': 'others',
    'other': 'others',
    'miscellaneous': 'others',
    
    # Rent
    'rent': 'rent',
    'Rent': 'rent',
    'RENT': 'rent',
    'housing': 'rent',
    
    # Savings
    'savings': 'savings',
    'Savings': 'savings',
    'SAVINGS': 'savings',
    'saving': 'savings',
    
    # Travel
    'travel': 'travel',
    'Travel': 'travel',
    'TRAVEL': 'travel',
    'transportation': 'travel',
    'transport': 'travel',
    
    # Utilities
    'utilities': 'utilities',
    'Utilities': 'utilities',
    'UTILITIES': 'utilities',
    'utilties': 'utilities',
    'Utilties': 'utilities',
    'utility': 'utilities'
}

# Income categories (map to 'income')
income_categories = [
    'salary', 'Salary', 'SALARY',
    'freelance', 'Freelance', 'FREELANCE', 
    'income', 'Income', 'INCOME',
    'business', 'Business', 'BUSINESS'
]

# Apply category mapping
def standardize_category(category):
    if pd.isna(category):
        return 'others'
    
    category_str = str(category).strip()
    
    # Check if it's an income category
    if category_str in income_categories:
        return 'income'
    
    # Check standard mappings
    if category_str in category_mapping:
        return category_mapping[category_str]
    
    # Default to others
    return 'others'

df['category_cleaned'] = df['category'].apply(standardize_category)

# 4. Standardize transaction types
print("4. Standardizing Transaction Types...")
df['transaction_type_cleaned'] = df['transaction_type'].str.lower().str.strip()
df.loc[df['category_cleaned'] == 'income', 'transaction_type_cleaned'] = 'income'
df.loc[df['category_cleaned'] != 'income', 'transaction_type_cleaned'] = 'expense'

# 5. Clean payment modes
print("5. Standardizing Payment Modes...")
def clean_payment_mode(mode):
    if pd.isna(mode):
        return 'unknown'
    
    mode_str = str(mode).lower().strip()
    
    if 'upi' in mode_str:
        return 'upi'
    elif 'card' in mode_str or 'crd' in mode_str:
        return 'card'
    elif 'cash' in mode_str or 'csh' in mode_str:
        return 'cash'
    elif 'bank' in mode_str or 'transfer' in mode_str:
        return 'bank_transfer'
    else:
        return 'others'

df['payment_mode_cleaned'] = df['payment_mode'].apply(clean_payment_mode)

# 6. Create additional features
print("6. Feature Engineering...")

# Extract date components
df['year'] = df['date_cleaned'].dt.year
df['month'] = df['date_cleaned'].dt.month
df['day'] = df['date_cleaned'].dt.day
df['day_of_week'] = df['date_cleaned'].dt.dayofweek  # 0=Monday, 6=Sunday
df['quarter'] = df['date_cleaned'].dt.quarter
df['week_of_year'] = df['date_cleaned'].dt.isocalendar().week

# Weekend flag
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Month start/end flags
df['is_month_start'] = df['date_cleaned'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date_cleaned'].dt.is_month_end.astype(int)

# Season mapping
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['month'].apply(get_season)

# 7. Create final cleaned dataset
print("7. Creating Final Dataset...")

# Select relevant columns
final_columns = [
    'transaction_id', 'user_id', 'date_cleaned', 'transaction_type_cleaned',
    'category_cleaned', 'amount_cleaned', 'payment_mode_cleaned', 'location',
    'notes', 'year', 'month', 'day', 'day_of_week', 'quarter', 'week_of_year',
    'is_weekend', 'is_month_start', 'is_month_end', 'season'
]

df_final = df[final_columns].copy()

# Rename columns for consistency
df_final.columns = [
    'transaction_id', 'user_id', 'date', 'transaction_type', 'category', 
    'amount', 'payment_mode', 'location', 'notes', 'year', 'month', 'day',
    'day_of_week', 'quarter', 'week_of_year', 'is_weekend', 'is_month_start',
    'is_month_end', 'season'
]

# Filter to realistic date range (2020-2025)
df_final = df_final[
    (df_final['year'] >= 2020) & 
    (df_final['year'] <= 2025)
]

# Remove extreme outliers in amounts
Q1 = df_final['amount'].quantile(0.25)
Q3 = df_final['amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR  # Using 3*IQR for more lenient outlier removal
upper_bound = Q3 + 3 * IQR

df_final = df_final[
    (df_final['amount'] >= max(0, lower_bound)) & 
    (df_final['amount'] <= upper_bound)
]

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")

# Display category distribution
print("\n=== CATEGORY DISTRIBUTION ===")
category_dist = df_final['category'].value_counts()
print(category_dist)

print("\n=== TRANSACTION TYPE DISTRIBUTION ===")
transaction_dist = df_final['transaction_type'].value_counts()
print(transaction_dist)

print("\n=== YEAR DISTRIBUTION ===")
year_dist = df_final['year'].value_counts().sort_index()
print(year_dist)

# Save cleaned dataset
df_final.to_csv('cleaned_real_dataset.csv', index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_real_dataset.csv'")

# Display sample of cleaned data
print("\n=== SAMPLE OF CLEANED DATA ===")
print(df_final.head(10))

print("\n=== DATA CLEANING COMPLETE ===")