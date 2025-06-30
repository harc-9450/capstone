
import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_and_scale_features(df):
    # One-hot encode categorical features
    categorical_cols = ['location', 'employment_status', 'job_roles', 'company_bucketed']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df = df.dropna()
    print(f"🧼 Cleaned features. Final dataset shape: {df.shape}")

    # Separate features and target
    X = df.drop('salary', axis=1)
    y = df['salary']

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns
