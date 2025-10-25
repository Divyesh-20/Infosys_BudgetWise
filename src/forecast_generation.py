# Real Dataset Forecasting and Budget Optimization
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=== PHASE 4: FORECASTING & BUDGET OPTIMIZATION (REAL DATA) ===")

# Load models and data
print("1. Loading Models and Data...")

# Load the best performing model (Random Forest)
with open('real_random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('real_gradient_boosting_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

with open('real_label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('real_feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load processed data
df = pd.read_csv('cleaned_real_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

print("✅ Models and data loaded successfully")

# Real Dataset Expense Forecasting Class
class RealDataExpenseForecastor:
    def __init__(self, model, data, encoders):
        self.model = model
        self.data = data
        self.encoders = encoders
        self.expense_data = data[data['transaction_type'] == 'expense'].copy()
        
        # Get feature columns
        self.feature_columns = [
            'category_encoded', 'payment_mode_encoded', 'location_encoded',
            'month', 'quarter', 'day_of_week', 'is_weekend', 
            'is_month_start', 'is_month_end', 'season_encoded'
        ]
        
        # Calculate average values for each category
        self.category_stats = self.expense_data.groupby('category').agg({
            'amount': ['mean', 'median', 'std', 'count'],
            'payment_mode': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'upi',
            'location': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mumbai'
        }).round(2)
        
        self.category_stats.columns = ['mean_amount', 'median_amount', 'std_amount', 'count', 'common_payment', 'common_location']
        
    def predict_category_expense(self, category, month, year=2025, user_context=None):
        """Predict expense for a specific category and month"""
        try:
            # Get category encoding
            if category in self.encoders['category'].classes_:
                category_encoded = self.encoders['category'].transform([category])[0]
            else:
                # Use most common category encoding if category not seen before
                category_encoded = 0
                
            # Get common payment mode for this category
            common_payment = self.category_stats.loc[category, 'common_payment'] if category in self.category_stats.index else 'upi'
            if common_payment in self.encoders['payment_mode'].classes_:
                payment_encoded = self.encoders['payment_mode'].transform([common_payment])[0]
            else:
                payment_encoded = 0
                
            # Get common location for this category  
            common_location = self.category_stats.loc[category, 'common_location'] if category in self.category_stats.index else 'Mumbai'
            
            # Handle location encoding (group less frequent locations)
            location_counts = self.expense_data['location'].value_counts()
            frequent_locations = location_counts[location_counts >= 50].index
            location_grouped = common_location if common_location in frequent_locations else 'Other_Location'
            
            if location_grouped in self.encoders['location'].classes_:
                location_encoded = self.encoders['location'].transform([location_grouped])[0]
            else:
                location_encoded = 0
                
            # Calculate other features
            quarter = (month - 1) // 3 + 1
            day_of_week = 1  # Assume Monday for prediction
            is_weekend = 0
            is_month_start = 0
            is_month_end = 0
            
            # Season encoding
            seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 
                      6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
            season = seasons[month]
            season_encoded = self.encoders['season'].transform([season])[0]
            
            # Create feature vector
            features = np.array([[
                category_encoded, payment_encoded, location_encoded,
                month, quarter, day_of_week, is_weekend,
                is_month_start, is_month_end, season_encoded
            ]])
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Ensure non-negative and reasonable prediction
            prediction = max(0, prediction)
            
            # If prediction seems unreasonable, fall back to historical average
            if category in self.category_stats.index:
                historical_mean = self.category_stats.loc[category, 'mean_amount']
                if prediction > historical_mean * 5 or prediction < historical_mean * 0.1:
                    prediction = historical_mean
                    
            return prediction
            
        except Exception as e:
            # Fallback to historical average if prediction fails
            if category in self.category_stats.index:
                return self.category_stats.loc[category, 'mean_amount']
            else:
                return 1000.0  # Default fallback
    
    def forecast_monthly_expenses(self, months_ahead=6, year=2025):
        """Forecast expenses for next N months"""
        current_date = datetime.now()
        forecasts = {}
        
        # Get all categories present in the data
        categories = self.expense_data['category'].unique()
        
        for i in range(1, months_ahead + 1):
            target_date = current_date + timedelta(days=30*i)
            target_month = target_date.month
            target_year = target_date.year
            
            month_forecast = {}
            total_forecast = 0
            
            for category in categories:
                if category != 'income':  # Skip income category for expense prediction
                    predicted_amount = self.predict_category_expense(category, target_month, target_year)
                    month_forecast[category] = round(predicted_amount, 2)
                    total_forecast += predicted_amount
            
            date_key = f"{target_year}-{target_month:02d}"
            forecasts[date_key] = {
                'categories': month_forecast,
                'total': round(total_forecast, 2)
            }
        
        return forecasts

# Real Dataset Budget Optimizer
class RealDataBudgetOptimizer:
    def __init__(self, data):
        self.data = data
        self.expense_data = data[data['transaction_type'] == 'expense'].copy()
        
    def get_historical_stats(self):
        """Get historical spending statistics by category"""
        stats = self.expense_data.groupby('category').agg({
            'amount': ['mean', 'median', 'std', 'min', 'max', 'sum', 'count']
        }).round(2)
        
        stats.columns = ['mean', 'median', 'std', 'min', 'max', 'total', 'count']
        return stats.reset_index()
    
    def suggest_budget_limits(self, income_amount, savings_goal=0.2):
        """Suggest budget limits for each category based on income and historical data"""
        # Calculate available budget after savings
        available_budget = income_amount * (1 - savings_goal)
        
        # Get historical spending statistics
        historical_stats = self.get_historical_stats()
        total_historical = historical_stats['total'].sum()
        
        # Calculate proportional budget allocation based on historical spending
        budget_suggestions = {}
        
        for _, row in historical_stats.iterrows():
            category = row['category']
            historical_proportion = row['total'] / total_historical
            suggested_monthly_budget = (available_budget * historical_proportion) * 12 / len(historical_stats)  # Monthly allocation
            
            # Get monthly historical average
            monthly_historical = row['total'] / (self.expense_data['year'].nunique() * 12)  # Approximate monthly average
            
            budget_suggestions[category] = {
                'suggested_monthly_limit': round(suggested_monthly_budget, 2),
                'historical_monthly_avg': round(monthly_historical, 2),
                'historical_median': round(row['median'], 2),
                'transaction_count': int(row['count']),
                'priority': self._get_category_priority(category)
            }
        
        return budget_suggestions
    
    def _get_category_priority(self, category):
        """Assign priority levels to categories"""
        priority_map = {
            'rent': 'High',
            'utilities': 'High', 
            'food': 'High',
            'health': 'High',
            'education': 'Medium',
            'travel': 'Medium',
            'investment': 'High',
            'savings': 'High',
            'entertainment': 'Low',
            'others': 'Low'
        }
        return priority_map.get(category, 'Medium')
    
    def analyze_spending_patterns(self):
        """Analyze spending patterns from real data"""
        patterns = {}
        
        # Monthly spending trends
        monthly_spending = self.expense_data.groupby(['year', 'month'])['amount'].sum()
        patterns['monthly_avg'] = monthly_spending.mean()
        patterns['monthly_std'] = monthly_spending.std()
        patterns['highest_month'] = monthly_spending.idxmax()
        patterns['lowest_month'] = monthly_spending.idxmin()
        
        # Seasonal patterns
        seasonal_spending = self.expense_data.groupby('season')['amount'].sum()
        patterns['highest_season'] = seasonal_spending.idxmax()
        patterns['seasonal_variation'] = seasonal_spending.std() / seasonal_spending.mean()
        
        # Category insights
        category_spending = self.expense_data.groupby('category')['amount'].sum()
        patterns['top_category'] = category_spending.idxmax()
        patterns['category_concentration'] = (category_spending.max() / category_spending.sum()) * 100
        
        return patterns

# Initialize forecasting and optimization
print("\n2. Initializing Real Data Forecasting System...")

forecaster = RealDataExpenseForecastor(rf_model, df, encoders)
optimizer = RealDataBudgetOptimizer(df)

# Generate forecasts
print("\n3. Generating 6-Month Expense Forecast...")
forecasts = forecaster.forecast_monthly_expenses(months_ahead=6)

print("=== REAL DATA EXPENSE FORECASTS ===")
for month, forecast in forecasts.items():
    print(f"\n{month}:")
    print(f"  Total Predicted: ₹{forecast['total']:,.2f}")
    print("  Category Breakdown:")
    for category, amount in sorted(forecast['categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {category}: ₹{amount:,.2f}")

# Historical analysis
print("\n4. Historical Spending Analysis...")
patterns = optimizer.analyze_spending_patterns()

print("=== SPENDING PATTERNS FROM REAL DATA ===")
print(f"Average Monthly Spending: ₹{patterns['monthly_avg']:,.2f}")
print(f"Monthly Spending Std Dev: ₹{patterns['monthly_std']:,.2f}")
print(f"Highest Spending Month: {patterns['highest_month'][1]}/{patterns['highest_month'][0]}")
print(f"Lowest Spending Month: {patterns['lowest_month'][1]}/{patterns['lowest_month'][0]}")
print(f"Highest Spending Season: {patterns['highest_season']}")
print(f"Top Spending Category: {patterns['top_category']}")
print(f"Category Concentration: {patterns['category_concentration']:.1f}% in top category")

# Budget optimization based on real data insights
print("\n5. Budget Optimization Based on Real Data...")

# Calculate realistic income from the data
income_transactions = df[df['transaction_type'] == 'income']['amount']
estimated_monthly_income = income_transactions.mean() if len(income_transactions) > 0 else 50000

print(f"Estimated Monthly Income (from data): ₹{estimated_monthly_income:,.2f}")

budget_suggestions = optimizer.suggest_budget_limits(estimated_monthly_income)

print("\n=== BUDGET SUGGESTIONS (REAL DATA INSIGHTS) ===")
print(f"Estimated Monthly Income: ₹{estimated_monthly_income:,.2f}")
print(f"Recommended Savings: 20% (₹{estimated_monthly_income * 0.2:,.2f})")
print(f"Available for Expenses: ₹{estimated_monthly_income * 0.8:,.2f}")

print("\nCategory-wise Budget Recommendations:")
for category, suggestion in sorted(budget_suggestions.items(), key=lambda x: x[1]['suggested_monthly_limit'], reverse=True):
    print(f"\n{category.upper()}:")
    print(f"  Suggested Monthly Limit: ₹{suggestion['suggested_monthly_limit']:,.2f}")
    print(f"  Historical Monthly Avg: ₹{suggestion['historical_monthly_avg']:,.2f}")
    print(f"  Transaction Count: {suggestion['transaction_count']}")
    print(f"  Priority: {suggestion['priority']}")

# Save results
print("\n6. Saving Real Data Forecasting Results...")

# Convert forecasts to DataFrame
forecast_data = []
for month, forecast in forecasts.items():
    for category, amount in forecast['categories'].items():
        forecast_data.append({
            'month': month,
            'category': category,
            'predicted_amount': amount
        })

forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_csv('real_data_expense_forecasts.csv', index=False)

# Save budget suggestions
budget_df = pd.DataFrame.from_dict(budget_suggestions, orient='index').reset_index()
budget_df.rename(columns={'index': 'category'}, inplace=True)
budget_df.to_csv('real_data_budget_suggestions.csv', index=False)

# Save spending patterns analysis
patterns_df = pd.DataFrame([patterns])
patterns_df.to_csv('real_data_spending_patterns.csv', index=False)

print("✅ Real data forecasting results saved!")

print("\n=== REAL DATA FORECASTING SUMMARY ===")
print(f"✅ Analyzed {len(df):,} real transactions")
print(f"✅ Generated forecasts for {len(forecasts)} months")  
print(f"✅ Budget suggestions for {len(budget_suggestions)} categories")
print(f"✅ Average monthly spending from data: ₹{patterns['monthly_avg']:,.2f}")
print(f"✅ Most expensive category: {patterns['top_category']}")
print(f"✅ Seasonal variation: {patterns['seasonal_variation']:.2f}")

print("\n=== REAL DATA FORECASTING COMPLETE ===")