import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

# Generate some sample data
X = np.random.rand(10, 5)  # Sample data matrix, 10 samples, 5 features

# Calculate pairwise cosine similarity
similarity_matrix = cosine_similarity(X)

print(similarity_matrix)