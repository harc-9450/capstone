import pandas as pd

def load_salary_dataset(path):
    df = pd.read_csv(path)

    # Normalize column names to lowercase with underscores
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # print("✅ Dataset loaded successfully.")
    # print(f"📊 Shape: {df.shape}\\n")

    # print("🧾 Columns:", df.columns.tolist(), "\\n")

    # print("🧼 Null values:")
    # print(df.isnull().sum(), "\\n")

    # print("📄 Sample data:")
    # print(df.head())

    return df
