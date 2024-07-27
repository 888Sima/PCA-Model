import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA with the number of components you want to keep
pca = PCA(n_components=2)

# Fit PCA on the standardized data and transform it
X_pca = pca.fit_transform(X_scaled)

# Print the results
print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
print("\nExplained variance ratio of each component:")
print(pca.explained_variance_ratio_)
print("\nPrincipal components:")
print(pca.components_)

# Create a DataFrame for better readability
df_pca = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Target'] = y
df_pca['Target Name'] = df_pca['Target'].apply(lambda x: target_names[x])

# Plotting the PCA results
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Target Name', data=df_pca, palette='viridis', s=100, alpha=0.7)

plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Target')
plt.grid(True)

# Show plot
plt.show()
