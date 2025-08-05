import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_r2_scores(results):
    model_names = list(results.keys())
    r2_scores = [results[m]['R2'] for m in model_names]
    
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=model_names, y=r2_scores, palette="crest", ax=ax)
    ax.set_title("R² Score Comparison")
    ax.set_ylim(0, 1)
    ax.set_ylabel("R² Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

def plot_true_vs_pred(y_true, y_pred, model_name, return_fig=False):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # 🔍 Higher DPI and bigger size

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", s=30)  # s = point size
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    
    ax.set_title(f"{model_name}: True vs Predicted", fontsize=14)
    ax.set_xlabel("True Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    if return_fig:
        return fig
    else:
        plt.show()
