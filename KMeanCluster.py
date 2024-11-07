import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Create synthetic dataset
n_samples = 500
n_features = 2
n_clusters = 5

X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# Convert to DataFrame for compatibility with your code
data = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data)

# Elbow method to find optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow method
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit KMeans with optimal clusters (set the optimal number of clusters as needed)
optimal_clusters = 5  # You can adjust this based on the elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(features_scaled)

# Predict and assign clusters to the original data
labels = kmeans.predict(features_scaled)
data['Cluster'] = labels

# Plotting the clusters
plt.figure(figsize=(10, 5))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis', label='Clusters')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print(data.head())
