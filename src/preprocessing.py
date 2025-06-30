
import pandas as pd

def preprocess_salary_data(df):
    # Drop columns we decided to ignore
    df = df.drop(columns=['salaries_reported'], errors='ignore')

    # Bucket company name
    top_companies = df['company_name'].value_counts().head(50).index.tolist()
    df['company_bucketed'] = df['company_name'].apply(lambda x: x if x in top_companies else "Other")

    # Drop original high-cardinality columns
    df = df.drop(columns=['company_name', 'job_title'], errors='ignore')

    return df
