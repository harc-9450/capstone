from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def apply_pca(X_scaled, n_components=3, feature_names=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print("📊 Explained Variance Ratios:", explained)
    print("🧮 Total Explained Variance:", round(sum(explained) * 100, 2), "%")

    if feature_names is not None:
        components_df = pd.DataFrame(pca.components_, columns=feature_names, index=[f"PC{i+1}" for i in range(n_components)])
        top_contributors = components_df.apply(lambda row: row.abs().nlargest(5).index.tolist(), axis=1)
        print("\n🔍 Top 5 contributing features per PCA component:")
        for pc, features in top_contributors.items():
            print(f"{pc}: {', '.join(features)}")

    return X_pca, explained

    return X_pca, explained

def plot_pca_2d(X_pca, y):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6)
    plt.xlabel("PC1 (Size & Layout Blend)")
    plt.ylabel("PC2 (Area Type + Balcony)")
    plt.title("2D PCA Projection Colored by Price")
    plt.colorbar(scatter, label="Price")
    plt.tight_layout()
    plt.show()

def plot_pca_3d(X_pca, y):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.6)
    ax.set_xlabel("PC1 (Size & Layout Blend)")
    ax.set_ylabel("PC2 (Area Type + Balcony)")
    ax.set_zlabel("PC3 (Location & Availability)")
    plt.title("3D PCA Projection (Colored by Price)")
    fig.colorbar(scatter)
    plt.show()
