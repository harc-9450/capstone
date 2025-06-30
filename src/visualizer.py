import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salaries")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30, color="purple")
    plt.title("Residual Distribution (Prediction Error)")
    plt.xlabel("Error in Salary Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
