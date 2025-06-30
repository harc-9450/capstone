import pandas as pd

def load_excel_as_documents(filepath: str):
    """Loads all sheets in an Excel file and flattens them into structured text + metadata."""
    xl = pd.ExcelFile(filepath)
    documents = []

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df.dropna(how="all", inplace=True)
        df = df.astype(str)

        for idx, row in df.iterrows():
            text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
            documents.append({
                "text": text,
                "metadata": {
                    "sheet": sheet_name,
                    "row": int(idx)
                }
            })

    return documents
