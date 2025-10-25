# Machine Learning Model Training on Real Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=== PHASE 3: MACHINE LEARNING MODEL TRAINING ===")

# Load cleaned dataset
df = pd.read_csv('cleaned_real_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

# Filter expense data for modeling (exclude income)
expense_data = df[df['transaction_type'] == 'expense'].copy()
print(f"Training data: {len(expense_data)} expense transactions")

# 1. Feature Engineering and Encoding
print("\n1. Feature Engineering...")

# Encode categorical variables
le_category = LabelEncoder()
le_payment = LabelEncoder()
le_location = LabelEncoder()
le_season = LabelEncoder()

expense_data['category_encoded'] = le_category.fit_transform(expense_data['category'])
expense_data['payment_mode_encoded'] = le_payment.fit_transform(expense_data['payment_mode'])

# Handle location with many unique values - group less frequent ones
location_counts = expense_data['location'].value_counts()
frequent_locations = location_counts[location_counts >= 50].index
expense_data['location_grouped'] = expense_data['location'].apply(
    lambda x: x if x in frequent_locations else 'Other_Location'
)
expense_data['location_encoded'] = le_location.fit_transform(expense_data['location_grouped'])

expense_data['season_encoded'] = le_season.fit_transform(expense_data['season'])

# 2. Prepare features for modeling
print("2. Preparing Feature Matrix...")

# Select features for modeling
feature_columns = [
    'category_encoded', 'payment_mode_encoded', 'location_encoded',
    'month', 'quarter', 'day_of_week', 'is_weekend', 
    'is_month_start', 'is_month_end', 'season_encoded'
]

X = expense_data[feature_columns].copy()
y = expense_data['amount'].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {feature_columns}")

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=expense_data['category']
)

# Scale features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 4. Model Evaluation Function
def evaluate_model(model, X_test, y_test, model_name, scaled=False):
    """Evaluate model performance"""
    X_eval = X_test_scaled if scaled else X_test
    y_pred = model.predict(X_eval)
    
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE safely (avoid division by zero)
    mask = y_test > 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else np.inf
    
    return {
        'Model': model_name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R²': round(r2, 4),
        'MAPE': round(mape, 2)
    }

# 5. Train Models
print("\n3. Training Machine Learning Models...")

model_results = []

# Linear Regression
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_results = evaluate_model(lr_model, X_test, y_test, 'Linear Regression', scaled=True)
model_results.append(lr_results)

# Decision Tree
print("Training Decision Tree...")
dt_model = DecisionTreeRegressor(random_state=42, max_depth=15, min_samples_split=20)
dt_model.fit(X_train, y_train)
dt_results = evaluate_model(dt_model, X_test, y_test, 'Decision Tree')
model_results.append(dt_results)

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42, max_depth=15, 
    min_samples_split=10, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_results = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
model_results.append(rf_results)

# Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=100, random_state=42, max_depth=8,
    learning_rate=0.1, min_samples_split=20
)
gb_model.fit(X_train, y_train)
gb_results = evaluate_model(gb_model, X_test, y_test, 'Gradient Boosting')
model_results.append(gb_results)

# 6. Display Results
print("\n=== MODEL PERFORMANCE COMPARISON ===")
results_df = pd.DataFrame(model_results)
print(results_df.to_string(index=False))

# 7. Feature Importance Analysis
print("\n4. Feature Importance Analysis...")

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Random Forest Feature Importance:")
print(rf_importance)

# Gradient Boosting feature importance
gb_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nGradient Boosting Feature Importance:")
print(gb_importance)

# 8. Category-Specific Models
print("\n5. Training Category-Specific Models...")

category_models = {}
category_results = []

# Get sufficient categories (at least 100 samples)
category_counts = expense_data['category'].value_counts()
sufficient_categories = category_counts[category_counts >= 100].index

print(f"Training category-specific models for {len(sufficient_categories)} categories")

for category in sufficient_categories:
    cat_data = expense_data[expense_data['category'] == category].copy()
    
    if len(cat_data) >= 100:  # Ensure sufficient data
        # Features excluding category (since it's constant)
        cat_features = [col for col in feature_columns if col != 'category_encoded']
        
        X_cat = cat_data[cat_features]
        y_cat = cat_data['amount']
        
        if len(X_cat) > 20:  # Minimum samples for split
            try:
                X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
                    X_cat, y_cat, test_size=0.2, random_state=42
                )
                
                # Train Random Forest for this category
                cat_model = RandomForestRegressor(
                    n_estimators=50, random_state=42, max_depth=10,
                    min_samples_split=5
                )
                cat_model.fit(X_cat_train, y_cat_train)
                
                # Evaluate
                cat_result = evaluate_model(cat_model, X_cat_test, y_cat_test, f'RF_{category}')
                category_results.append(cat_result)
                category_models[category] = {
                    'model': cat_model,
                    'features': cat_features,
                    'samples': len(cat_data)
                }
                
            except Exception as e:
                print(f"Error training model for {category}: {e}")

print(f"Successfully trained {len(category_models)} category-specific models")

# Display category-specific results
if category_results:
    print("\n=== CATEGORY-SPECIFIC MODEL PERFORMANCE ===")
    cat_results_df = pd.DataFrame(category_results)
    print(cat_results_df.to_string(index=False))

# 9. Model Selection and Validation
print("\n6. Model Selection...")

best_model_idx = results_df['MAPE'].idxmin()
best_model_name = results_df.iloc[best_model_idx]['Model']
best_mape = results_df.iloc[best_model_idx]['MAPE']
best_r2 = results_df.iloc[best_model_idx]['R²']

print(f"Best Overall Model: {best_model_name}")
print(f"MAPE: {best_mape}%")
print(f"R²: {best_r2}")

# Select best model object
model_mapping = {
    'Linear Regression': lr_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}

best_model = model_mapping[best_model_name]

# 10. Sample Predictions
print("\n7. Sample Predictions...")

# Make predictions with best model
if best_model_name == 'Linear Regression':
    sample_predictions = best_model.predict(X_test_scaled[:10])
else:
    sample_predictions = best_model.predict(X_test[:10])

sample_predictions = np.maximum(sample_predictions, 0)  # Ensure non-negative
actual_values = y_test.iloc[:10].values

prediction_comparison = pd.DataFrame({
    'Actual': actual_values,
    'Predicted': sample_predictions.round(2),
    'Error': (actual_values - sample_predictions).round(2),
    'Error_%': ((actual_values - sample_predictions) / actual_values * 100).round(2)
})

print("Sample Predictions vs Actual:")
print(prediction_comparison)

# 11. Save Models and Results
print("\n8. Saving Models and Results...")

# Save main models
with open('real_linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
    
with open('real_random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
    
with open('real_gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

with open('real_decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

# Save preprocessing objects
with open('real_feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('real_label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'category': le_category,
        'payment_mode': le_payment,
        'location': le_location,
        'season': le_season
    }, f)

# Save category models
with open('real_category_models.pkl', 'wb') as f:
    pickle.dump(category_models, f)

# Save results
results_df.to_csv('real_model_performance_results.csv', index=False)

if category_results:
    cat_results_df.to_csv('real_category_model_results.csv', index=False)

# Save feature importance
rf_importance.to_csv('real_rf_feature_importance.csv', index=False)
gb_importance.to_csv('real_gb_feature_importance.csv', index=False)

print("✅ Models and results saved successfully!")

print(f"\n=== MODELING RESULTS SUMMARY ===")
print(f"✅ Dataset: {len(expense_data):,} expense transactions")
print(f"✅ Features: {len(feature_columns)} engineered features")
print(f"✅ Models trained: {len(model_results)} general + {len(category_models)} category-specific")
print(f"✅ Best model: {best_model_name} (MAPE: {best_mape}%, R²: {best_r2})")
print(f"✅ Feature importance: {rf_importance.iloc[0]['feature']} is most important ({rf_importance.iloc[0]['importance']:.3f})")

print("\n=== MACHINE LEARNING PHASE COMPLETE ===")