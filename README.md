# ⚙️ AutoML Streamlit App

An interactive and no-code AutoML tool built with **Streamlit** that allows users to:

- 📥 Upload a CSV dataset
- 🔍 Perform EDA (Exploratory Data Analysis)
- 🧠 Train multiple classification or regression models
- 📊 Evaluate model performance
- 🏆 Download the best-performing model (based on test metrics)

---

## 🚀 Features

- Auto-detects classification vs. regression tasks
- Supports common models like:
  - Linear, Ridge, Lasso, ElasticNet
  - Decision Trees, Random Forest, Gradient Boosting
  - KNN, XGBoost, LightGBM, SVR
- Hyperparameter tuning via GridSearchCV
- Interactive plots and profiling
- Downloadable `.pkl` file of the best model

---

## 🛠️ How to Run

> ⚠️ Before you begin: Make sure you have Python 3.7+ and `pip` installed.

1. **Clone the repository**  
   ```bash
   git clone https://github.com/vamshi-M96/AutoML.git
   cd AutoML
