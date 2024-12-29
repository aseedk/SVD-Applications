import pprint

import numpy as np

from Calculate_SVD import calculate_svd

# Function to make predictions for the recommendation system
def recommend(matrix, rank):
    """Perform SVD on the matrix and make recommendations."""
    # Compute SVD
    U, S, Vt = calculate_svd(matrix)

    U = U[:, :rank]
    S = np.diag(S[:rank])
    Vt = Vt[:rank, :]

    # Reconstruct the matrix from U, S, and Vt
    reconstructed_matrix = np.dot(np.dot(U, S), Vt)

    # Predictions for the missing values
    return reconstructed_matrix


# Given input matrix A
A = np.array([
    [2, 1],
    [1, 0],
    [0, 1]
], dtype=np.float64)

# Set the rank for the SVD decomposition (number of latent factors)
rank = 1

# Get recommendations by performing SVD
predicted_matrix = recommend(A, rank)

# Display the original matrix and the predicted ratings (reconstructed matrix)
print("Original Matrix A:")
pprint.pprint(A)

print("\nReconstructed Matrix (Predicted Ratings):")
pprint.pprint(predicted_matrix)