
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

def plot_pca_regression_surface(X_pca_3, y, title="PCA 3D Regression Surface"):
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_pca_3, y)

    # Create meshgrid across first two PCA components
    x_range = np.linspace(X_pca_3[:, 0].min(), X_pca_3[:, 0].max(), 30)
    y_range = np.linspace(X_pca_3[:, 1].min(), X_pca_3[:, 1].max(), 30)
    xx, yy = np.meshgrid(x_range, y_range)
    zz_input = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]

    # Create z by predicting across the grid (fill 3rd component with avg value or 0)
    zz_input[:, 2] = np.mean(X_pca_3[:, 2])  # fixed third component for a 2D plane slice
    zz_pred = model.predict(zz_input).reshape(xx.shape)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the regression surface
    ax.plot_surface(xx, yy, zz_pred, color="cyan", alpha=0.5, edgecolor="none")

    # Plot actual points
    ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], color="red", label="Actual", alpha=0.6)

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
