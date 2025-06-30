from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_on_pca_features(X_pca, y, n_components=3):

    X_reduced = X_pca[:, :n_components]
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📊 PCA-based Regression using {n_components} components")
    print(f"🔢 RMSE: {rmse:.2f}")
    print(f"📈 R² Score: {r2:.3f}")

    return model, rmse, r2
