from sklearn.preprocessing import StandardScaler
import pandas as pd

def encode_and_scale_features(df):
    # One-hot encode area_type
    area_dummies = pd.get_dummies(df['area_type'], drop_first=True)
    df = pd.concat([df.drop('area_type', axis=1), area_dummies], axis=1)

    # One-hot encode location
    location_dummies = pd.get_dummies(df['location'], drop_first=True)
    df = pd.concat([df.drop('location', axis=1), location_dummies], axis=1)

    # Drop any remaining NaNs
    df = df.dropna()
    print(f"🧼 Cleaned features. Final dataset shape: {df.shape}")

    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Ensure only numeric features go into the scaler
    X = X.select_dtypes(include=[float, int, bool])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns 
