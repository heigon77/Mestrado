import numpy as np

# Load the concatenated matrix
concatenated_matrix = np.load("concatenated_matrix.npy")

# Get the dimensions of the matrix
n = concatenated_matrix.shape[0]

# Copy the values from the upper triangle to the lower triangle
for i in range(n):
    for j in range(i + 1, n):
        concatenated_matrix[j, i] = concatenated_matrix[i, j]

# Save the modified matrix to a new .npy file
np.save("concatenated_matrix_complete.npy", concatenated_matrix)
