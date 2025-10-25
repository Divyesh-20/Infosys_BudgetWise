
# 💰 Infosys BudgetWise - AI-Powered Personal Finance Management

An intelligent personal finance system leveraging AI to forecast expenses, optimize budgets, and provide insightful spending analysis. Designed for ease of use and actionable insights, BudgetWise helps you take control of your financial future.

🎯 **Why Infosys BudgetWise is Special**

BudgetWise goes beyond simple tracking, offering:

*   🔮 **AI-Powered Forecasting:** Predict your future expenses with accuracy, looking 6-12 months ahead.
*   💡 **Smart Budget Recommendations:** Receive personalized budget suggestions based on your unique spending patterns and financial goals.
*   📊 **Deep Dive Analytics:** Uncover hidden insights into your spending habits, category trends, and monthly financial behavior.
*   🎛️ **Interactive Dashboard:** A user-friendly Streamlit interface for real-time visualization and in-depth analysis of your financial data.

🚀 **Quick Start Guide**

### 🖥️ Prerequisites

*   Python 3.8 or higher
*   At least 4GB of RAM (for optimal model performance)
*   A modern web browser (Chrome, Firefox, Safari, Edge)

### ⚡ Installation & Setup

**Method 1: Standard Setup (Recommended)**



**Method 2: Alternative Setup (Manual Dependency Installation)**

If you prefer to manage dependencies manually:

**Method 3: Using Docker (Containerized Deployment)**

For a consistent and isolated environment, use Docker:

> 1.  Ensure you have Docker installed on your system.
> 2.  Build the Docker image: `docker build -t budgetwise .`
> 3.  Run the Docker container: `docker run -p 8501:8501 budgetwise`
> 4.  Access the dashboard in your browser at `http://localhost:8501`

📊 **Dashboard Features**

*   📈 **Financial Overview Dashboard**
    *   Real-time display of key financial metrics derived from your transaction data.
    *   Interactive pie charts illustrating category-wise spending distribution.
    *   Visualizations of monthly trends and seasonal patterns in your expenses.
    *   In-depth analysis of your spending behavior and financial habits.
*   🔍 **Data-Driven Insights**
    *   Detailed segmentation of spending by category for easy identification of spending hotspots.
    *   Analysis of seasonal trends in expenses across different months of the year.
    *   Insights into your preferred payment methods (UPI, Card, Cash, Bank Transfer).
    *   Location-based spending insights for travel and local expenses.
*   💰 **Intelligent Budget Planning**
    *   AI-driven personalized budget recommendations tailored to your income and spending habits.
    *   Custom budget calculator to help you plan for savings and investments.
    *   Priority-based budget allocation across different spending categories.
    *   Tools for planning and managing your emergency fund.
*   🔮 **Predictive Expense Forecasting**
    *   AI-powered forecasting of expenses for the next 6 months, with a detailed category-wise breakdown.
    *   Multiple forecasting methods (Historical Average, Seasonal Adjustment, and Machine Learning-based).
    *   Interactive charts with drill-down capabilities to explore forecast details.
    *   Confidence intervals to assess the reliability of expense predictions.
*   📤 **Seamless Data Upload**
    *   Effortlessly upload your transaction data in CSV format.
    *   Automatic data cleaning and validation to ensure data quality.
    *   Instant dashboard analysis and visualization of your uploaded data.

🛠️ **Technical Architecture**

1.  **Raw Data:** Ingestion of transaction data from CSV or other sources.
2.  **Cleaning & Validation:** Data cleaning, handling missing values, and validating data types.
3.  **Feature Engineering:** Creation of new features from existing data for model training.
4.  **Model Training:** Training machine learning models (Random Forest, XGBoost, Linear Regression) to forecast expenses.
5.  **Forecasting & Budget Optimization:** Using trained models to predict future expenses and optimize budget allocation.
6.  **Interactive Streamlit Dashboard:** Visualization of data insights and forecasts in an interactive dashboard.

🤖 **ML Models Overview**

| Model             | Use Case                                 | Description                                                                                                                               |
| ----------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Random Forest     | Primary Expense Forecasting              | An ensemble learning method that operates by constructing multiple decision trees and outputting the mode of the classes (regression) of the individual trees. |
| XGBoost           | Feature Importance & Accuracy Enhancement | Optimized gradient boosting algorithm used to enhance the accuracy of expense predictions and identify key features influencing spending. |
| Linear Regression | Baseline Comparison                      | A linear approach for modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). Used as a baseline for comparing the performance of more complex models. |

🔑 **Engineered Features**

*   **Temporal Features:** Month, Quarter, Weekday, Weekend to capture time-based patterns.
*   **Seasonal Features:** Summer, Winter, Monsoon to identify seasonal spending variations.
*   **Categorical Features:** Payment mode, Category, Location to segment spending by type and place.
*   **Behavioral Features:** Start/end month spending trends to capture changes in spending behavior over time.

💡 **Insights & Data Highlights**

*   **Transactions:** Synthetic dataset emulating real-world spending patterns across various categories.
>   You can also upload your real-world transaction data to get personalized insights.
*   **Categories:** 🏠 Rent, 🍕 Food, 💡 Utilities, 🏥 Health, ✈️ Travel, 🎓 Education, 🎬 Entertainment, 💰 Investments, 💾 Savings, 💼 Others
*   **User behavior:** Track patterns over months to identify changes in spending habits.
*   **Payment trends:** Predominantly digital (UPI, Card) reflecting the shift towards cashless transactions.

🏷️ **Sample Category Overview**

python
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load your training data
data = pd.read_csv('training_data.csv')

# Preprocess your data (example: handling missing values, encoding categorical features)
data = data.fillna(data.mean())  # Replace missing values with the mean
data = pd.get_dummies(data, columns=['category', 'payment_mode'])  # One-hot encode categorical features

# Define your features (X) and target (y)
X = data.drop('amount', axis=1)
y = data['amount']

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Now, load the saved model for predictions:
with open('rf_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Prepare your new data for prediction
new_data = pd.read_csv('new_data.csv')
new_data = new_data.fillna(new_data.mean())
new_data = pd.get_dummies(new_data, columns=['category', 'payment_mode'])

# Ensure the new data has the same columns as the training data
missing_cols = set(X.columns) - set(new_data.columns)
for c in missing_cols:
    new_data[c] = 0
new_data = new_data[X.columns]

# Make predictions
predictions = loaded_model.predict(new_data)
print(predictions)
python
# Example of a simple budget optimization function
def suggest_budget_limits(monthly_income, savings_rate):
    discretionary_income = monthly_income * (1 - savings_rate)
    # Example allocations (can be customized)
    rent = discretionary_income * 0.3
    food = discretionary_income * 0.2
    utilities = discretionary_income * 0.1
    savings = monthly_income * savings_rate
    
    optimized_budget = {
        'Rent': rent,
        'Food': food,
        'Utilities': utilities,
        'Savings': savings,
        'Discretionary': discretionary_income * 0.4  # Remaining for other expenses
    }
    return optimized_budget

monthly_income = 75000
savings_rate = 0.20
optimized_budget = suggest_budget_limits(monthly_income, savings_rate)
print(optimized_budget)
def preprocess_transactions(df):
    # Example preprocessing steps (customize as needed)
    df['date'] = pd.to_datetime(df['date'])
    df = df.fillna(0)  # Handle missing values
    # Add more cleaning and feature engineering steps here
    return df

def generate_forecasts(cleaned_df):
    # Placeholder for forecast generation (replace with your model)
    forecasts = cleaned_df.groupby('category')['amount'].mean()
    return forecasts

# Load your transaction data
df = pd.read_csv('your_transactions.csv')

# Preprocess the data
cleaned = preprocess_transactions(df)



📈 **Performance**

*   Dashboard load time: Under 3 seconds
*   Model inference time: Less than 100ms
*   Optimized for smooth visualization with 10K+ transactions

🔍 **Troubleshooting**

*   **Dashboard not launching:**
    *   Ensure Streamlit is correctly installed: `streamlit --version`
    *   Upgrade Streamlit to the latest version: `pip install --upgrade streamlit`
    *   Check for port conflicts (default port is 8501).  If another application is using this port, try running Streamlit with a different port: `streamlit run app.py --server.port 8080`
*   **Missing model files:**
    *   Verify that all `.pkl` model files are located in the project root directory.
    *   If the files are missing, re-train the models using the provided scripts.
*   **CSV upload issues:**
    *   Ensure your CSV file has the correct columns: `date`, `amount`, `category`, `payment_mode`.
    *   Verify the date format is `YYYY-MM-DD`.
    *   Check for any special characters or inconsistencies in the data that may cause parsing errors.
    *   Use a tool like Notepad++ (Windows) or Sublime Text (macOS/Linux) to ensure the file is encoded in UTF-8.
*   **Model giving unexpected results:**
    *   Ensure your input data is preprocessed in the same way as the training data.
    *   Check for data leakage (features that contain information about the target variable).
    *   Review the feature engineering steps to ensure the features are relevant and properly scaled.

🤝 **Contributing**

We welcome contributions to enhance Infosys BudgetWise! Here are some areas where you can contribute:

*   **Model Improvement & Hyperparameter Tuning:** Experiment with different machine learning models and optimize hyperparameters to improve forecasting accuracy.
*   **Feature Engineering & New Insights:** Develop new features to capture additional patterns in the data and provide more insightful analysis.
*   **UI/UX Enhancements:** Improve the user interface and user experience of the Streamlit dashboard to make it more intuitive and user-friendly.
*   **Integration with Banking APIs:** Integrate with banking APIs to automatically fetch transaction data and eliminate the need for manual data uploads.
*   **Deployment Optimization:** Optimize the deployment process to make it easier to deploy Infosys BudgetWise on different platforms.
>   If you're interested in contributing, please fork the repository and submit a pull request.  Be sure to follow our coding conventions and include detailed documentation for your changes.

📝 **License**

MIT License - see the [LICENSE](LICENSE) file for details.

🙏 **Acknowledgments**

*   Open-source libraries: Streamlit, Pandas, Scikit-learn, Plotly
*   The machine learning and dashboard community for inspiration and best practices.

📞 **Support**

*   📧 Email: your-email@domain.com
*   🐛 Issues: Create a GitHub issue for bug reports or feature requests.

🎯 **Start exploring your AI-powered personal finance dashboard:**

