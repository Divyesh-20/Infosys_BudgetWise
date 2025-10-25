# Exploratory Data Analysis on Real Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load cleaned dataset
df = pd.read_csv('cleaned_real_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

print("=== PHASE 2: EXPLORATORY DATA ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# 1. Basic Statistics
print("\n1. Dataset Overview:")
print(f"Total transactions: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total amount: ₹{df['amount'].sum():,.2f}")

# 2. Category Analysis
print("\n2. Category Analysis:")
expense_data = df[df['transaction_type'] == 'expense'].copy()
income_data = df[df['transaction_type'] == 'income'].copy()

category_stats = expense_data.groupby('category').agg({
    'amount': ['count', 'sum', 'mean', 'median', 'std']
}).round(2)

category_stats.columns = ['count', 'total', 'mean', 'median', 'std']
category_stats = category_stats.sort_values('total', ascending=False)

print("Category Statistics (Expenses Only):")
print(category_stats)

# 3. Monthly Trends
print("\n3. Monthly Spending Trends:")
monthly_expenses = expense_data.groupby(['year', 'month']).agg({
    'amount': 'sum'
}).reset_index()

monthly_expenses['date'] = pd.to_datetime(monthly_expenses[['year', 'month']].assign(day=1))
monthly_expenses = monthly_expenses.sort_values('date')

print("Monthly Expense Totals:")
print(monthly_expenses[['year', 'month', 'amount']].head(10))

# 4. User Analysis
print("\n4. User Spending Analysis:")
user_stats = expense_data.groupby('user_id').agg({
    'amount': ['count', 'sum', 'mean'],
    'category': lambda x: x.nunique()
}).round(2)

user_stats.columns = ['transaction_count', 'total_spent', 'avg_amount', 'categories_used']
user_stats = user_stats.sort_values('total_spent', ascending=False)

print("Top 10 Spenders:")
print(user_stats.head(10))

# 5. Seasonal Analysis
print("\n5. Seasonal Spending Patterns:")
seasonal_stats = expense_data.groupby(['season', 'category']).agg({
    'amount': ['sum', 'mean']
}).round(2)

seasonal_stats.columns = ['total', 'average']
seasonal_spending = expense_data.groupby('season')['amount'].sum().sort_values(ascending=False)

print("Total Spending by Season:")
print(seasonal_spending)

# 6. Payment Mode Analysis
print("\n6. Payment Mode Analysis:")
payment_stats = df.groupby('payment_mode').agg({
    'amount': ['count', 'sum', 'mean']
}).round(2)

payment_stats.columns = ['transactions', 'total_amount', 'avg_amount']
payment_stats = payment_stats.sort_values('total_amount', ascending=False)

print("Payment Mode Statistics:")
print(payment_stats)

# 7. Weekend vs Weekday Analysis
print("\n7. Weekend vs Weekday Spending:")
weekend_analysis = expense_data.groupby('is_weekend').agg({
    'amount': ['count', 'sum', 'mean']
}).round(2)

weekend_analysis.columns = ['transactions', 'total_amount', 'avg_amount']
weekend_analysis.index = ['Weekday', 'Weekend']

print("Weekend vs Weekday Comparison:")
print(weekend_analysis)

# 8. Location Analysis
print("\n8. Location Analysis:")
location_stats = df.groupby('location').agg({
    'amount': ['count', 'sum', 'mean']
}).round(2)

location_stats.columns = ['transactions', 'total_amount', 'avg_amount']
location_stats = location_stats.sort_values('total_amount', ascending=False)

print("Top 10 Locations by Total Amount:")
print(location_stats.head(10))

# 9. Create aggregated data for modeling
print("\n9. Creating Monthly Aggregated Data for Modeling...")

# Monthly aggregation by user and category
monthly_user_category = df.groupby(['user_id', 'year', 'month', 'category']).agg({
    'amount': 'sum'
}).reset_index()

monthly_user_category['date'] = pd.to_datetime(monthly_user_category[['year', 'month']].assign(day=1))

print(f"Monthly user-category data shape: {monthly_user_category.shape}")
print("Sample of monthly aggregated data:")
print(monthly_user_category.head())

# Overall monthly aggregation
monthly_totals = df.groupby(['year', 'month', 'category']).agg({
    'amount': 'sum'
}).reset_index()

monthly_totals['date'] = pd.to_datetime(monthly_totals[['year', 'month']].assign(day=1))

print(f"\nOverall monthly totals shape: {monthly_totals.shape}")

# 10. Target categories check
print("\n10. Target Categories Verification:")
required_categories = [
    'education', 'entertainment', 'food', 'health', 'income', 
    'investment', 'others', 'rent', 'savings', 'travel', 'utilities'
]

available_categories = df['category'].unique()
print("Required categories:", required_categories)
print("Available categories:", sorted(available_categories))

missing_categories = set(required_categories) - set(available_categories)
extra_categories = set(available_categories) - set(required_categories)

print(f"Missing categories: {missing_categories if missing_categories else 'None'}")
print(f"Extra categories: {extra_categories if extra_categories else 'None'}")

# Check if we have sufficient data for each category
print("\nData sufficiency by category (min 50 transactions recommended):")
category_counts = df['category'].value_counts()
for cat in required_categories:
    if cat in category_counts:
        count = category_counts[cat]
        status = "✅ Sufficient" if count >= 50 else "⚠️ Limited"
        print(f"{cat}: {count} transactions - {status}")
    else:
        print(f"{cat}: 0 transactions - ❌ Missing")

# Save processed data
monthly_user_category.to_csv('monthly_user_category_data.csv', index=False)
monthly_totals.to_csv('monthly_totals_data.csv', index=False)

print("\n✅ EDA Complete. Files saved:")
print("- cleaned_real_dataset.csv")
print("- monthly_user_category_data.csv") 
print("- monthly_totals_data.csv")

print(f"\n=== DATASET READY FOR MODELING ===")
print(f"✅ {len(df):,} clean transactions")
print(f"✅ {df['user_id'].nunique()} users")
print(f"✅ {len(available_categories)} categories")
print(f"✅ {len(monthly_totals)} monthly data points")
print(f"✅ Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

print("\n=== EDA PHASE COMPLETE ===")