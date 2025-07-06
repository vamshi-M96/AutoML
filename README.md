# ⚙️ AutoML Streamlit App

An interactive **no-code AutoML tool** built with Streamlit that allows users to:

- 📥 Upload datasets (CSV/XLSX)
- 🔍 Perform Exploratory Data Analysis (EDA)
- 🧠 Train multiple machine learning models (Classification or Regression)
- 📊 Evaluate model performance with accuracy, R², RMSE, etc.
- 🏆 Automatically select and download the best-performing model
- 🔮 Make predictions (manually or in batch)

---

## 🚀 Features

✅ Auto-detects task type (Classification vs. Regression)  
✅ Built-in models:
- **Regression**: Linear, Ridge, Lasso, ElasticNet, SVR, RandomForest, XGBoost, LightGBM
- **Classification**: Logistic Regression, RandomForest, KNN, SVC, Naive Bayes, XGBoost, LGBM, Bagging, Decision Tree

✅ Model tuning via `GridSearchCV` & `RandomizedSearchCV`  
✅ PCA support for dimensionality reduction  
✅ Visualizations: Histograms, Boxplots, Correlation Heatmaps, Countplots, Pairplots  
✅ Predictive Power Score (PPS) & Sweetviz Report  
✅ Download `.pkl` file of best model  
✅ Manual and batch prediction with encoded/decoded outputs

---

## 🛠️ How to Run Locally

> ⚠️ Requires Python 3.7+ and pip

### 1. Clone the repository
```bash
git clone https://github.com/vamshi-M96/AutoML.git
cd AutoML
