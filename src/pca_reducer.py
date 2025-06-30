
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def apply_pca(X_scaled, n_components=3, feature_names=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print("📊 Explained Variance Ratios:", explained)
    print("🧮 Total Explained Variance:", round(sum(explained) * 100, 2), "%")

    component_labels = [f"PC{i+1}" for i in range(n_components)]
    axis_labels = component_labels.copy()

    if feature_names is not None:
        components_df = pd.DataFrame(pca.components_, columns=feature_names, index=component_labels)
        print("\n🔍 Top 5 contributing features per PCA component:")
        for i, pc in enumerate(components_df.index):
            sorted_features = components_df.loc[pc].abs().sort_values(ascending=False)
            top_features = sorted_features.head(5).index.tolist()
            axis_labels[i] = f"{pc}: " + ", ".join(top_features[:2])  # short axis label

            print(f"{pc} (Top contributors): {', '.join(top_features)}")
            print("   ↳ Weighted contributions:")
            for feat in top_features:
                weight = components_df.loc[pc, feat]
                print(f"     • {feat}: {weight:.4f}")
            print("")

    return X_pca, explained, axis_labels

def plot_pca_2d(X_pca, y, axis_labels=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6)
    plt.xlabel(axis_labels[0] if axis_labels else "PC1")
    plt.ylabel(axis_labels[1] if axis_labels else "PC2")
    plt.title("2D PCA Projection Colored by Salary")
    plt.colorbar(scatter, label="Salary")
    plt.tight_layout()
    plt.show()

def plot_pca_3d(X_pca, y, axis_labels=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.6)
    ax.set_xlabel(axis_labels[0] if axis_labels else "PC1")
    ax.set_ylabel(axis_labels[1] if axis_labels else "PC2")
    ax.set_zlabel(axis_labels[2] if axis_labels else "PC3")
    plt.title("3D PCA Projection (Colored by Salary)")
    fig.colorbar(scatter)
    plt.show()
