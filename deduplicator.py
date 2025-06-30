import pandas as pd

def detect_duplicates(csv_path, subset_columns=None):
    df = pd.read_csv(csv_path)

    # Find duplicates (keep=False marks all duplicates, not just later ones)
    if subset_columns:
        duplicates = df[df.duplicated(subset=subset_columns, keep=False)]
    else:
        duplicates = df[df.duplicated(keep=False)]

    return df, duplicates

def remove_duplicates(df, subset_columns=None):
    cleaned = df.drop_duplicates(subset=subset_columns, keep="first")
    return cleaned

def save_to_csv(df, output_path):
    df.to_csv(output_path, index=False)