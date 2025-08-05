# 🔬 streamlit-regression-comparison-app

A Streamlit-based interactive application to **compare regression models** on the **Life Expectancy dataset**. It includes data cleaning, training multiple models, visualizing results, and evaluating performance using metrics and plots.

---

## 📌 Features

✅ Clean and preprocess the Life Expectancy dataset  
✅ Train and evaluate multiple regression models  
✅ Visualize model performance using R² scores and prediction accuracy  
✅ Interactive Streamlit interface for dataset upload, preview, and model comparison  
✅ Modular and extensible codebase

---

## 📁 Project Structure

```
regression_comparison_app/
│
├── app.py                      # Main Streamlit app
├── clean.py                    # Data cleaning script
├── models.py                   # Train and evaluate regression models
├── utils.py                    # Plotting and helper functions
│
├── data/
│   └── Life_Expectancy_Data.csv     # Original dataset
│   └── life_expectancy_cleaned.csv # Cleaned version (auto-generated)
│
├── images/                    # Optional: Exported plots (if saved)
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview (this file)
```

---

## 📊 Models Implemented

| Model                               | Description                                   |
| ----------------------------------- | --------------------------------------------- |
| **Linear Regression**               | Simple linear fit to data                     |
| **Polynomial Regression**           | Degree 2 regression with feature scaling      |
| **Support Vector Regression (SVR)** | Scaled + kernelized regression                |
| **Decision Tree**                   | Tree-based, non-linear partitioning           |
| **Random Forest**                   | Ensemble of decision trees (boosted accuracy) |

---

## 📈 Visualizations

- **📄 Dataset Preview:** Top 5 rows of the dataset
- **📊 Metric Table:** MSE and R² for all models
- **📈 R² Score Bar Chart:** Visual comparison of model R² scores
- **📉 True vs Predicted Scatter Plot:** For each model, shows prediction performance

---

## 🚀 How to Run

1. **Clone the repo**

```bash
git clone https://github.com/anianimesh905/streamlit-regression-comparison-app.git
cd streamlit-regression-comparison-app
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app.py
```

---

## 🧪 Upload Your Own Dataset

Want to try it on your own data?

1. Go to the Streamlit sidebar
2. Uncheck **"Use default dataset"**
3. Upload your CSV file

⚠️ **Note:** Your file should include a continuous target column like `Life expectancy`.

---

## 🧹 Data Cleaning Steps

- Drops irrelevant columns (`Country`, `Year`, `Status`)
- Removes rows with too many missing values
- Drops zero-variance columns
- Saves a cleaned CSV for model training

---

## 💡 Insights

- **SVR** and **Random Forest** gave the best performance (highest R²)
- **Polynomial Regression** may overfit if features are not scaled
- **Decision Tree** results can appear smooth due to many feature splits, but behavior is piecewise constant (step-like)

---

## 📦 Dependencies

```txt
streamlit
pandas
scikit-learn
matplotlib
seaborn
```

Install all via:

```bash
pip install -r requirements.txt
```

---

## 📸 Sample Screenshot

> _You can include a sample screenshot from your app UI (optional)_

---

## 📚 Dataset Source

[Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

> The dataset includes socio-economic and health data from WHO reports.

---

## 📂 To-Do / Extensions

- [ ] Add Ridge/Lasso Regression
- [ ] Export model reports as PDF
- [ ] Deploy on Streamlit Cloud
- [ ] Add feature importance visualizations

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- Dataset: World Health Organization (via Kaggle)
- Inspired by regression modeling best practices
