# app.py
# Streamlit app — Modernized (Dark Theme, Tabbed Navigation)
# NOTE: UI changes only. Core logic and function names are preserved verbatim.

import os
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
import plotly.express as px

# ---------------------------
# Dark theme CSS (UI-only)
# ---------------------------
st.set_page_config(page_title="BudgetWise ", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    /* Dark background */
    html, body, [class*="css"] {
        background: #0b1220 !important;
        color: #e6eef8 !important;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    /* Header */
    .main-header {
        font-size: 2.2rem;
        color: #66b2ff;
        text-align: left;
        margin-bottom: 0.2rem;
        font-weight: 700;
    }
    .sub-header {
        color: #a9c9ff;
        margin-top: -6px;
        margin-bottom: 14px;
        font-size: 0.95rem;
    }
    /* Card */
    .card {
        background: linear-gradient(180deg, rgba(16,24,40,0.85), rgba(12,18,30,0.85));
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.04);
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
    }
    .metric-card {
        background: linear-gradient(180deg, rgba(12,18,30,0.85), rgba(10,14,24,0.85));
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #1f6feb, #5aa0ff);
        color: #031025;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border: none;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(37,114,255,0.14);
    }
    /* Sidebar collapsed text */
    section[data-testid="stSidebar"] { display: none; }
    /* Tables */
    .stDataFrame, .stTable {
        background: transparent;
        color: #e6eef8;
    }
    /* Footer */
    .footer {
        color: #9fb7ff;
        text-align: center;
        padding: 0.6rem;
        font-size: 0.9rem;
    }
    /* Plotly container */
    .plotly-chart { border-radius: 10px; padding: 8px; background: rgba(255,255,255,0.02); }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Header (UI-only)
# ---------------------------
col1, col2 = st.columns([7, 3])
with col1:
    st.markdown('<div class="main-header">💼 BudgetWise </div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Expense Forecasting · Budget Optimization · Operational Insights</div>', unsafe_allow_html=True)
# with col2:
     #st.markdown(f"<div style='text-align:right; color:#bcd6ff; font-weight:600;'>Last run: {datetime.now().strftime('%b %d, %Y %H:%M')}</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# --- CORE LOGIC (PRESERVED) ---
# The functions below are copied from your original script and kept intact.
# Do NOT change names: load_real_models_and_data(), predict_real_data_expenses()
# ---------------------------
@st.cache_data
def load_real_models_and_data():
    """Load real data models and datasets"""
    try:
        # --- Define base directories relative to this file ---
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')

        # --- Load models ---
        with open(os.path.join(models_dir, 'real_random_forest_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        with open(os.path.join(models_dir, 'real_gradient_boosting_model.pkl'), 'rb') as f:
            gb_model = pickle.load(f)
        with open(os.path.join(models_dir, 'real_label_encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)

        # --- Load data ---
        df = pd.read_csv(os.path.join(data_dir, 'raw/cleaned_real_dataset.csv'))
        df['date'] = pd.to_datetime(df['date'])

        # --- Load analysis results ---
        forecasts = pd.read_csv(os.path.join(data_dir, 'processed/real_data_expense_forecasts.csv'))
        budget_suggestions = pd.read_csv(os.path.join(data_dir, 'processed/real_data_budget_suggestions.csv'))
        spending_patterns = pd.read_csv(os.path.join(data_dir, 'processed/real_data_spending_patterns.csv'))

        return rf_model, gb_model, encoders, df, forecasts, budget_suggestions, spending_patterns

    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        return None, None, None, None, None, None, None

def predict_real_data_expenses(model, data, encoders, months_ahead=6):
    """Predict expenses using real data patterns"""
    expense_data = data[data['transaction_type'] == 'expense'].copy()
    categories = expense_data['category'].unique()

    # Calculate category averages for fallback
    category_stats = expense_data.groupby('category')['amount'].agg(['mean', 'median', 'std']).round(2)

    forecasts = {}
    current_date = datetime.now()

    for i in range(1, months_ahead + 1):
        target_date = current_date + timedelta(days=30*i)
        target_month = target_date.month
        target_year = target_date.year

        month_forecast = {}

        for category in categories:
            if category != 'income':
                try:
                    # Use statistical approach based on historical data
                    if category in category_stats.index:
                        base_amount = category_stats.loc[category, 'mean']

                        # Add seasonal variation
                        seasonal_multiplier = 1.0
                        if target_month in [6, 7, 8]:  # Summer
                            seasonal_multiplier = 1.1 if category in ['travel', 'entertainment'] else 0.9
                        elif target_month in [11, 12, 1]:  # Winter/Holiday
                            seasonal_multiplier = 1.2 if category in ['food', 'entertainment', 'others'] else 1.0

                        predicted_amount = base_amount * seasonal_multiplier
                        month_forecast[category] = max(0, predicted_amount)
                    else:
                        month_forecast[category] = 1000.0  # Default

                except Exception:
                    month_forecast[category] = category_stats.loc[category, 'mean'] if category in category_stats.index else 1000.0

        date_key = f"{target_year}-{target_month:02d}"
        forecasts[date_key] = month_forecast

    return forecasts

# ---------------------------
# --- END CORE LOGIC (PRESERVED) ---
# ---------------------------

# Load models and data (kept as-is)
with st.spinner("Loading models and datasets..."):
    rf_model, gb_model, encoders, df, forecasts_df, budget_df, patterns_df = load_real_models_and_data()

if df is None:
    st.error("❌ Unable to load data. Please ensure all data files are available.")
    st.stop()

# ---------------------------
# Tabbed Navigation (UI-only)
# ---------------------------
tab_names = ["📈 Dashboard", "🔍 Insights", "🔮 Forecasting", "💰 Budget", "📊 Models", "📤 Upload"]
tabs = st.tabs(tab_names)

# ---------------------------
# Tab 1: Dashboard
# ---------------------------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 📈 Financial Dashboard — Real Data")
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"**Date Range:** {df['date'].min().strftime('%B %Y')} — {df['date'].max().strftime('%B %Y')}")
        st.markdown(f"**Total Transactions:** {len(df):,}")
        st.markdown(f"**Unique Users:** {df['user_id'].nunique()}")
        st.markdown(f"**Categories:** {df['category'].nunique()}")
    with col2:
        total_amount = df['amount'].sum()
        st.metric("💸 Total Transaction Volume", f"₹{total_amount:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("💰 Key Financial Metrics")
    expense_data = df[df['transaction_type'] == 'expense']
    income_data = df[df['transaction_type'] == 'income']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_expenses = expense_data['amount'].sum()
        st.metric("Total Expenses", f"₹{total_expenses:,.0f}")
    with col2:
        total_income = income_data['amount'].sum()
        st.metric("Total Income", f"₹{total_income:,.0f}")
    with col3:
        avg_transaction = expense_data['amount'].mean()
        st.metric("Avg Transaction", f"₹{avg_transaction:,.0f}")
    with col4:
        monthly_avg = expense_data.groupby([expense_data['date'].dt.year, expense_data['date'].dt.month])['amount'].sum().mean()
        st.metric("Avg Monthly Spend", f"₹{monthly_avg:,.0f}")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💸 Expense Distribution by Category")
        category_totals = expense_data.groupby('category')['amount'].sum().sort_values(ascending=False)
        fig_pie = px.pie(values=category_totals.values, names=category_totals.index, title="Expense Distribution")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("📊 Top 5 Categories by Amount")
        top_categories = category_totals.head()
        fig_bar = px.bar(x=top_categories.values, y=top_categories.index, orientation='h', title="Top Spending Categories")
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📈 Monthly Spending Trends")
    expense_data = expense_data.copy()
    expense_data['year'] = expense_data['date'].dt.year
    expense_data['month'] = expense_data['date'].dt.month
    monthly_expenses = expense_data.groupby(['year', 'month'])['amount'].sum().reset_index()
    monthly_expenses['date'] = pd.to_datetime(monthly_expenses[['year', 'month']].assign(day=1))
    fig_line = px.line(monthly_expenses, x='date', y='amount', title="Monthly Expense Trends", markers=True)
    fig_line.update_layout(xaxis_title="Month", yaxis_title="Amount (₹)")
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Tab 2: Insights (UI-only wraps existing logic)
# ---------------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🔍 Deep Data Analysis & Insights")
    expense_data = df[df['transaction_type'] == 'expense']

    # User analysis
    st.subheader("👥 User Spending Analysis")
    col1, col2 = st.columns(2)
    with col1:
        user_stats = expense_data.groupby('user_id').agg({'amount': ['count', 'sum', 'mean']}).round(2)
        user_stats.columns = ['transactions', 'total_spent', 'avg_amount']
        user_stats = user_stats.sort_values('total_spent', ascending=False)
        st.write("**Top 10 Users by Total Spending:**")
        st.dataframe(user_stats.head(10))
    with col2:
        user_totals = user_stats['total_spent']
        fig_hist = px.histogram(x=user_totals, nbins=20, title="Distribution of User Total Spending")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Seasonal analysis
    st.subheader("🌍 Seasonal Spending Patterns")
    if 'season' in expense_data.columns:
        seasonal_data = expense_data.groupby(['season', 'category'])['amount'].sum().reset_index()
        fig_seasonal = px.bar(seasonal_data, x='season', y='amount', color='category', title="Seasonal Spending by Category", barmode='stack')
        st.plotly_chart(fig_seasonal, use_container_width=True)
    else:
        st.info("Season column not present in dataset. Skipping seasonal analysis.")

    # Payment mode analysis
    st.subheader("💳 Payment Mode Analysis")
    col1, col2 = st.columns(2)
    with col1:
        payment_stats = df.groupby('payment_mode').agg({'amount': ['count', 'sum', 'mean']}).round(2)
        payment_stats.columns = ['transactions', 'total_amount', 'avg_amount']
        st.dataframe(payment_stats)
    with col2:
        st.subheader("📅 Weekend vs Weekday Spending")
        if 'is_weekend' in expense_data.columns:
            weekend_data = expense_data.groupby('is_weekend')['amount'].agg(['count', 'sum', 'mean']).round(2)
            weekend_data.index = ['Weekday', 'Weekend']
            weekend_data.columns = ['Transactions', 'Total', 'Average']
            st.dataframe(weekend_data)
        else:
            st.info("is_weekend column not present in dataset. Skipping weekend vs weekday analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Tab 3: Forecasting
# ---------------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🔮 Expense Forecasting")
    col1, col2 = st.columns([1, 2])
    with col1:
        months_ahead = st.slider("📅 Forecast Period (Months)", min_value=1, max_value=12, value=6)
        forecast_method = st.selectbox("🔬 Forecasting Method", ["Historical Average", "Seasonal Adjusted", "Machine Learning"])
    with col2:
        st.markdown("""
        **Forecasting Methods**
        - **Historical Average**: Based on past spending averages
        - **Seasonal Adjusted**: Considers seasonal spending patterns
        - **Machine Learning**: Uses trained Random Forest model
        """)
    if st.button("🚀 Generate Forecast", key="generate_forecast"):
        with st.spinner("Generating forecasts..."):
            forecasts = predict_real_data_expenses(rf_model, df, encoders, months_ahead)
            st.subheader("📊 Expense Forecasts")

            forecast_data = []
            for month, categories in forecasts.items():
                for category, amount in categories.items():
                    forecast_data.append({'month': month, 'category': category, 'predicted_amount': amount})

            forecast_df = pd.DataFrame(forecast_data)
            monthly_totals = forecast_df.groupby('month')['predicted_amount'].sum().reset_index()
            monthly_totals['month'] = pd.to_datetime(monthly_totals['month'])

            fig_forecast = px.bar(monthly_totals, x='month', y='predicted_amount', title="Monthly Total Expense Forecast")
            fig_forecast.update_layout(xaxis_title="Month", yaxis_title="Predicted Amount (₹)")
            st.plotly_chart(fig_forecast, use_container_width=True)

            fig_stacked = px.bar(forecast_df, x='month', y='predicted_amount', color='category', title="Category-wise Expense Forecast", barmode='stack')
            st.plotly_chart(fig_stacked, use_container_width=True)

            st.subheader("📋 Detailed Monthly Forecasts")
            pivot_forecast = forecast_df.pivot(index='category', columns='month', values='predicted_amount').fillna(0).round(0)
            st.dataframe(pivot_forecast.style.format("₹{:,.0f}"))

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Tab 4: Budget Planning
# ---------------------------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("💰 Smart Budget Planning")
    st.subheader("📋 AI-Powered Budget Recommendations")
    st.markdown("Based on your historical spending patterns:")

    # Budget display (fallback if budget_df missing)
    try:
        budget_display = budget_df.copy()
        budget_display = budget_display.sort_values('suggested_monthly_limit', ascending=False)
        fig_budget = px.bar(budget_display, x='category', y='suggested_monthly_limit', color='priority',
                            title="Recommended Monthly Budget Limits by Category")
        st.plotly_chart(fig_budget, use_container_width=True)
    except Exception:
        st.info("Budget suggestions file not available or malformed.")

    st.subheader("🎯 Custom Budget Planner")
    col1, col2 = st.columns(2)
    with col1:
        monthly_income = st.number_input("💵 Your Monthly Income (₹)", value=50000, step=5000)
        savings_rate = st.slider("💾 Savings Rate (%)", min_value=10, max_value=50, value=20)
    with col2:
        investment_rate = st.slider("📈 Investment Rate (%)", min_value=5, max_value=30, value=10)
        emergency_months = st.slider("🚨 Emergency Fund (Months)", min_value=3, max_value=12, value=6)

    savings_amount = monthly_income * (savings_rate / 100)
    investment_amount = monthly_income * (investment_rate / 100)
    available_for_expenses = monthly_income - savings_amount - investment_amount
    emergency_fund_target = available_for_expenses * emergency_months

    st.subheader("💰 Your Personalized Budget")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monthly Income", f"₹{monthly_income:,.0f}")
    with col2:
        st.metric("Savings", f"₹{savings_amount:,.0f}")
    with col3:
        st.metric("Investment", f"₹{investment_amount:,.0f}")
    with col4:
        st.metric("Available for Expenses", f"₹{available_for_expenses:,.0f}")

    allocation_data = {'Category': ['Expenses', 'Savings', 'Investment'], 'Amount': [available_for_expenses, savings_amount, investment_amount]}
    fig_allocation = px.pie(allocation_data, values='Amount', names='Category', title=f"Monthly Income Allocation (₹{monthly_income:,.0f})")
    st.plotly_chart(fig_allocation, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)



# Tab 5: Model Performance
# ---------------------------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Machine Learning Model Performance")

    # --- Define base/project directories ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    processed_dir = os.path.join(base_dir, 'data/processed')
    features_dir = os.path.join(base_dir, 'data/features')

    # --- Load and display model performance ---
    performance_file = os.path.join(processed_dir, 'real_model_performance_results.csv')
    if os.path.exists(performance_file):
        model_results = pd.read_csv(performance_file)
        st.subheader("🏆 Model Comparison")
        st.dataframe(model_results)

        # Plot performance
        fig_performance = px.bar(
            model_results, 
            x='Model', 
            y='MAPE', 
            title="Model Performance Comparison (Lower MAPE is Better)"
        )
        st.plotly_chart(fig_performance, use_container_width=True)
    else:
        st.error("Model performance data not available.")

    # --- Load and display feature importance ---
    feature_file = os.path.join(features_dir, 'real_rf_feature_importance.csv')
    if os.path.exists(feature_file):
        rf_importance = pd.read_csv(feature_file)
        fig_importance = px.bar(
            rf_importance, 
            x='importance', 
            y='feature', 
            orientation='h', 
            title="Random Forest Feature Importance"
        )
        fig_importance.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("Feature importance data not available.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Tab 6: Upload New Data
# ---------------------------
with tabs[5]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📤 Upload Your Financial Data")
    st.markdown("""
    ### 📋 Upload Guidelines
    Upload your transaction data in CSV format with the following columns:

    **Required Columns:**
    - `date`: Transaction date (YYYY-MM-DD, DD/MM/YYYY, or similar)
    - `amount`: Transaction amount (numeric)
    - `category`: Expense category
    - `transaction_type`: 'Income' or 'Expense'

    **Optional Columns:**
    - `user_id`, `payment_mode`, `location`, `notes`
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.subheader("📊 Your Data Preview")
            st.dataframe(user_df.head())
            st.subheader("🔍 Data Quality Check")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- Rows: {len(user_df):,}")
                st.write(f"- Columns: {len(user_df.columns)}")
                st.write(f"- Missing values: {user_df.isnull().sum().sum()}")
            with col2:
                st.write("**Columns Found:**")
                for col in user_df.columns:
                    st.write(f"- {col}")

            if 'amount' in user_df.columns and 'category' in user_df.columns:
                user_df['amount_clean'] = pd.to_numeric(user_df['amount'].astype(str).str.replace('[₹Rs,.]', '', regex=True), errors='coerce')
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_amount = user_df['amount_clean'].sum()
                    st.metric("Total Amount", f"₹{total_amount:,.0f}")
                with col2:
                    avg_transaction = user_df['amount_clean'].mean()
                    st.metric("Avg Transaction", f"₹{avg_transaction:,.0f}")
                with col3:
                    transaction_count = len(user_df)
                    st.metric("Total Transactions", f"{transaction_count:,}")

                if user_df['category'].nunique() > 1:
                    category_totals = user_df.groupby('category')['amount_clean'].sum().sort_values(ascending=False)
                    fig = px.pie(values=category_totals.values, names=category_totals.index, title="Your Expense Distribution")
                    st.plotly_chart(fig, use_container_width=True)

            if st.button("🔬 Analyze with AI Models", key="analyze_uploaded"):
                st.info("🚀 This feature would integrate your data with our trained models for personalized insights!")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Footer (UI-only)
# ---------------------------
st.markdown("---")
st.markdown("<div class='footer'>🚀 BudgetWise Analytics · Built for professional finance teams · © 2025</div>", unsafe_allow_html=True)
