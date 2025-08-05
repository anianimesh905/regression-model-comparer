import streamlit as st
import pandas as pd
from clean import clean_life_expectancy_dataset
from models import train_all_models
from utils import plot_r2_scores, plot_true_vs_pred

# Page config
st.set_page_config(page_title="Life Expectancy Regression Comparison", layout="wide")
st.title("🔬 Life Expectancy - Regression Model Comparison")

# Load and clean default dataset directly
data_path = "data/Life_Expectancy_Data.csv"
cleaned_path = clean_life_expectancy_dataset(data_path)
df = pd.read_csv(cleaned_path)
st.success("✅ Default dataset cleaned and loaded successfully.")


# Show preview
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Model Training
st.subheader("🧠 Training Regression Models")
with st.spinner("Training..."):
    results, y_tests, y_preds = train_all_models(df)

# Display evaluation results
st.subheader("📊 Evaluation Metrics")
st.dataframe(pd.DataFrame(results).T.round(3))

# R² Score Plot
st.subheader("📈 R² Score Comparison")
plot_r2_scores(results)

# True vs Predicted for each model
st.subheader("📉 True vs Predicted Comparison")
for model_name in y_preds:
    with st.expander(f"📌 {model_name}"):
        fig = plot_true_vs_pred(y_tests[model_name], y_preds[model_name], model_name, return_fig=True)
        st.pyplot(fig)
