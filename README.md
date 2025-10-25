---

# 💰 Infosys BudgetWise - Personal Expense Forecasting & Budget Optimization

> AI-powered personal finance management system using **synthetic and real-like transaction data** for predictive insights, budgeting, and spending analysis.

---

## 🎯 Why Infosys BudgetWise is Special

This isn’t just a tracker. It’s a **smart finance companion**:

* **🔮 AI Forecasting** – Predict future expenses up to 6–12 months
* **💡 Smart Budget Recommendations** – Personalized suggestions based on spending patterns
* **📊 Deep Analytics** – Insights on category trends, payment modes, and monthly behavior
* **🎛️ Interactive Dashboard** – Streamlit UI for real-time visualization and analysis

---

## 🚀 Quick Start Guide

### 🖥️ Prerequisites

* Python 3.8+
* 4GB RAM (for model usage)
* Modern web browser

### ⚡ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Divyesh-20/Infosys_BudgetWise.git
cd Infosys_BudgetWise

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py

# 4. Open in your browser
# Navigate to: http://localhost:8501
```

### 🔄 Alternative Setup

```bash
pip install streamlit pandas numpy scikit-learn plotly
streamlit run app.py
```

---

## 📊 Dashboard Features

### 📈 Financial Dashboard

* **Real-time metrics** from all transactions
* **Category distribution** with interactive pie charts
* **Monthly trends** & seasonal pattern visualization
* **User behavior analysis**

### 🔍 Data Insights

* Spending segmentation by category
* Seasonal trends across months
* Payment mode preferences: **UPI, Card, Cash, Bank Transfer**
* Location-wise insights

### 💰 Budget Planning

* AI-powered **personalized recommendations**
* Custom **budget calculator** for savings & investments
* **Priority-based allocation** for categories
* **Emergency fund planning**

### 🔮 Expense Forecasting

* **6-month predictions** with category-wise breakdown
* Multiple forecasting methods: **Historical, Seasonal, ML-based**
* Interactive charts with drill-down capabilities
* Confidence intervals for prediction reliability

### 📤 Data Upload

* Upload **your own CSV** of transactions
* Automatic cleaning and validation
* Instant dashboard analysis

---

## 🛠️ Technical Architecture

### 🔄 Data Pipeline

```python
Raw Data
   ↓
Cleaning & Validation
   ↓
Feature Engineering
   ↓
Model Training (Random Forest, XGBoost, Linear Regression)
   ↓
Forecasting & Budget Optimization
   ↓
Interactive Streamlit Dashboard
```

### 🤖 ML Models Overview

| Model                | Use Case                      |
| -------------------- | ----------------------------- |
| 🌳 Random Forest     | Primary forecasting           |
| ⚡ XGBoost            | Feature importance & accuracy |
| 📐 Linear Regression | Baseline comparison           |

### 🔑 Engineered Features

* **Temporal**: Month, Quarter, Weekday, Weekend
* **Seasonal**: Summer, Winter, Monsoon
* **Categorical**: Payment mode, Category, Location
* **Behavioral**: Start/end month spending trends

---

## 💡 Insights & Data Highlights

* **Transactions**: Synthetic dataset emulating real spending patterns
* **Categories**: 🏠 Rent, 🍕 Food, 💡 Utilities, 🏥 Health, ✈️ Travel, 🎓 Education, 🎬 Entertainment, 💰 Investments, 💾 Savings, 💼 Others
* **User behavior**: Track patterns over months
* **Payment trends**: Predominantly digital (UPI, Card)

### 🏷️ Sample Category Overview

| Category         | Transactions | Avg Amount | Priority |
| ---------------- | ------------ | ---------- | -------- |
| 🏠 Rent          | 1,200        | ₹7,500     | High     |
| 🍕 Food          | 2,400        | ₹4,000     | High     |
| 💡 Utilities     | 800          | ₹4,800     | High     |
| 🏥 Health        | 400          | ₹4,900     | High     |
| ✈️ Travel        | 1,000        | ₹5,100     | Medium   |
| 🎓 Education     | 450          | ₹5,400     | Medium   |
| 🎬 Entertainment | 550          | ₹5,000     | Low      |
| 💰 Investment    | 100          | ₹20,000    | High     |
| 💾 Savings       | 270          | ₹5,300     | High     |
| 💼 Others        | 4,200        | ₹6,300     | Low      |

---

## 🔧 Advanced Usage

### 🔮 Custom Forecasting

```python
import pickle
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(features_df)
```

### 💵 Budget Optimization

```python
monthly_income = 75000
savings_rate = 0.20
optimized_budget = optimizer.suggest_budget_limits(monthly_income, savings_rate)
```

### 📂 Data Integration

```python
import pandas as pd
df = pd.read_csv('your_transactions.csv')
cleaned = preprocess_transactions(df)
forecasts = generate_forecasts(cleaned)
```

---

## 📈 Performance

* Dashboard **load time**: <3 seconds
* **Model inference**: <100ms
* Smooth visualization for **10K+ transactions**

---

## 🔍 Troubleshooting

**Dashboard not launching:**

```bash
streamlit --version
pip install --upgrade streamlit
```

**Missing model files:** Ensure all `.pkl` files are in project root

**CSV upload issues:** Columns: `date`, `amount`, `category`, `payment_mode` (Date: YYYY-MM-DD)

---

## 🤝 Contributing

Contributions welcome!

* Model improvement & hyperparameter tuning
* Feature engineering & new insights
* UI/UX enhancements
* Integration with banking APIs
* Deployment optimization

---

## 📝 License

MIT License – see LICENSE file.

---

## 🙏 Acknowledgments

* Open-source libraries: Streamlit, Pandas, Scikit-learn, Plotly
* ML & Dashboard community for inspiration & best practices

---


```bash
pip install -r requirements.txt
streamlit run app.py
```

Access your dashboard at: [http://localhost:8501](http://localhost:8501)

---
to do that next?

