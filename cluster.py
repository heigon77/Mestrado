import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

# Generate some sample data
# X = np.random.rand(10000, 5)  # Sample data matrix, 10 samples, 5 features


# Calculate pairwise Euclidean distances
distance_matrix = np.load('concatenated_matrix_complete.npy')

distance_matrix /= 100

# Convert distances to similarities
similarity_matrix = 1 / (1 + distance_matrix)

# print(similarity_matrix)

damping = 0.5  # Decrease the damping factor (default is 0.5)
preference = -100  # Adjust the preference parameter based on your data
# Create an instance of AffinityPropagation with adjusted parameters
affinity_propagation = AffinityPropagation(affinity='precomputed', damping=damping)


# Fit the model to the precomputed similarity matrix
affinity_propagation.fit(similarity_matrix)

# Get cluster labels
cluster_labels = affinity_propagation.labels_

# Get exemplars
exemplars = affinity_propagation.cluster_centers_indices_

# Count the number of elements per cluster
cluster_counts = Counter(cluster_labels)

# Sort clusters by the number of elements (highest first)
sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)

with open('cluster_info.txt', 'w') as file:
    file.write("Number of elements per cluster (sorted in descending order):\n")
    for cluster, count in sorted_clusters:
        file.write(f"Cluster {cluster}: {count} elements\n")
