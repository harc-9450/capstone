import sys
sys.path.append('src')
from pca_3d_plotter import plot_pca_regression_surface
from pca_reducer import apply_pca, plot_pca_2d, plot_pca_3d
from visualizer import plot_actual_vs_predicted, plot_residuals
from model_builder import train_and_evaluate_model
from data_laoder import load_salary_dataset
from preprocessing import preprocess_salary_data
from feature_engineering import encode_and_scale_features

# Load and preprocess data
df = load_salary_dataset("data/Salary Dataset.csv")
df_cleaned = preprocess_salary_data(df)
X_scaled, y, feature_names = encode_and_scale_features(df_cleaned)

# Train and evaluate full regression model
model, X_test, y_test, y_pred = train_and_evaluate_model(X_scaled, y)
plot_actual_vs_predicted(y_test, y_pred)
plot_residuals(y_test, y_pred)

# PCA with 2 components for 2D projection
X_pca_2, explained_2, axis_labels_2 = apply_pca(X_scaled, n_components=2, feature_names=feature_names)
plot_pca_2d(X_pca_2, y, axis_labels=axis_labels_2)

# PCA with 3 components for 3D projection
X_pca_3, explained_3, axis_labels_3 = apply_pca(X_scaled, n_components=3, feature_names=feature_names)
plot_pca_2d(X_pca_3, y, axis_labels=axis_labels_2)
plot_pca_3d(X_pca_3, y, axis_labels=axis_labels_3)

print("\\n📦 Training model with PCA (2 components)...")
model_pca_2, X_test_2, y_test_2, y_pred_2 = train_and_evaluate_model(X_pca_2, y)
plot_actual_vs_predicted(y_test_2, y_pred_2)
plot_residuals(y_test_2, y_pred_2)

print("\\n📦 Training model with PCA (3 components)...")
model_pca_3, X_test_3, y_test_3, y_pred_3 = train_and_evaluate_model(X_pca_3, y)
plot_actual_vs_predicted(y_test_3, y_pred_3)
plot_residuals(y_test_3, y_pred_3)

plot_pca_regression_surface(X_pca_3, y, title="PCA 3D Regression Surface")