import pandas as pd

def load_bangalore_housing_data(path):
    df = pd.read_csv(path)
    print("✅ Dataset loaded successfully.")
    print("📊 Shape:", df.shape)
    print("\n🧾 Columns:", df.columns.tolist())
    print("\n🧼 Null values:\n", df.isnull().sum())
    return df