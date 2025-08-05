import pandas as pd
import os

def clean_life_expectancy_dataset(
    input_path="data/Life_Expectancy_Data.csv",
    output_path="data/life_expectancy_cleaned.csv"
):
    # Step 1: Load dataset
    print(f"📥 Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    original_shape = df.shape
    print(f"Original shape: {original_shape}")

    # Step 2: Drop non-numeric or irrelevant columns
    drop_cols = ["Country", "Year", "Status"]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)
    print(f"Dropped columns: {drop_cols}")

    # Step 3: Drop rows with more than 3 missing values
    df.dropna(thresh=len(df.columns) - 3, inplace=True)
    print(f"Shape after dropping sparse rows: {df.shape}")

    # Step 4: Drop any remaining missing values
    df.dropna(inplace=True)
    print(f"Shape after dropping remaining NaNs: {df.shape}")

    # Step 5: Drop columns with no variance
    zero_var_cols = df.columns[df.nunique() <= 1].tolist()
    df = df.loc[:, df.nunique() > 1]
    print(f"Dropped zero-variance columns: {zero_var_cols}")

    # Step 6: Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")

    return output_path
