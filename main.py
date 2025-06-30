import sys
sys.path.append("src")
from pca_3d_plotter import plot_pca_regression_surface
from data_loader import load_bangalore_housing_data
from preprocessing import preprocess_bangalore_data
from feature_engineering import encode_and_scale_features
from model_builder import train_and_evaluate_model
from visualizer import plot_actual_vs_predicted, plot_residuals
from pca_reducer import apply_pca, plot_pca_2d, plot_pca_3d
from pca_model_comparison import train_on_pca_features

# Load and preprocess
df = load_bangalore_housing_data("data/Bengaluru_House_Data.csv")
df_cleaned = preprocess_bangalore_data(df)
X_scaled, y, feature_names = encode_and_scale_features(df_cleaned)

# Full model training
model, X_test, y_test, y_pred = train_and_evaluate_model(X_scaled, y)
plot_actual_vs_predicted(y_test, y_pred)
plot_residuals(y_test, y_pred)

# PCA (2 components) for 2D plot and regression
X_pca_2, explained_2 = apply_pca(X_scaled, n_components=2, feature_names=feature_names)
plot_pca_2d(X_pca_2, y)

# PCA (3 components) for 3D plot and regression
X_pca_3, explained_3 = apply_pca(X_scaled, n_components=3, feature_names=feature_names)
plot_pca_3d(X_pca_3, y)




# Evaluate with PCA (2)
model_pca2, rmse_pca2, r2_pca2 = train_on_pca_features(X_pca_2, y, n_components=2)

# Evaluate with PCA (3)
model_pca3, rmse_pca3, r2_pca3 = train_on_pca_features(X_pca_3, y, n_components=3)

# Final visual: Actual vs Predicted for PCA(2)
model_pca_viz2, X_test_pca2, y_test_pca2, y_pred_pca2 = train_and_evaluate_model(X_pca_2, y)
plot_actual_vs_predicted(y_test_pca2, y_pred_pca2)
plot_residuals(y_test_pca2, y_pred_pca2)

# Final visual: Actual vs Predicted for PCA(3)
model_pca_viz3, X_test_pca3, y_test_pca3, y_pred_pca3 = train_and_evaluate_model(X_pca_3, y)
plot_actual_vs_predicted(y_test_pca3, y_pred_pca3)
plot_residuals(y_test_pca3, y_pred_pca3)

plot_pca_regression_surface(X_pca_3, y, title="PCA 3D Regression Surface")