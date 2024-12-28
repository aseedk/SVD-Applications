import numpy as np


# Function to compute SVD without libraries
def compute_svd(matrix, rank):
    """Compute the Singular Value Decomposition (SVD) of a matrix."""
    AAT = np.dot(matrix, matrix.T)  # A * A^T
    ATA = np.dot(matrix.T, matrix)  # A^T * A

    # Compute eigenvalues and eigenvectors
    U_eigenvalues, U = np.linalg.eigh(AAT)  # Eigenvectors for AAT
    V_eigenvalues, V = np.linalg.eigh(ATA)  # Eigenvectors for ATA

    # Sort eigenvalues in descending order and sort corresponding eigenvectors
    U_indices = np.argsort(U_eigenvalues)[::-1]
    V_indices = np.argsort(V_eigenvalues)[::-1]

    U = U[:, U_indices]
    V = V[:, V_indices]

    # Compute singular values
    singular_values = np.sqrt(np.maximum(U_eigenvalues[U_indices], 0))  # Non-negative values

    # Select the top `rank` singular values and corresponding vectors
    U = U[:, :rank]
    S = np.diag(singular_values[:rank])
    Vt = V.T[:rank, :]

    return U, S, Vt


# Function to make predictions for the recommendation system
def recommend(matrix, rank):
    """Perform SVD on the matrix and make recommendations."""
    # Compute SVD
    U, S, Vt = compute_svd(matrix, rank)

    # Reconstruct the matrix from U, S, and Vt
    reconstructed_matrix = np.dot(np.dot(U, S), Vt)

    # Predictions for the missing values
    return reconstructed_matrix


# Given input matrix A
A = np.array([
    [2, 1],
    [1, 0],
    [0, 1]
])

# Set the rank for the SVD decomposition (number of latent factors)
rank = 2

# Get recommendations by performing SVD
predicted_matrix = recommend(A, rank)

# Display the original matrix and the predicted ratings (reconstructed matrix)
print("Original Matrix A:")
print(A)

print("\nReconstructed Matrix (Predicted Ratings):")
print(predicted_matrix)


B = np.array([[5, 4, np.nan, 2, np.nan],
                       [3, np.nan, np.nan, 1, 4],
                       [4, np.nan, 5, 3, 4],
                       [1, 1, 3, np.nan, 5],
                       [2, 3, np.nan, 4, 3]])
B = np.nan_to_num(B, nan=0)
# Set the rank for the SVD decomposition (number of latent factors)
rank = 2

# Get recommendations by performing SVD
predicted_matrix = recommend(B, rank)

# Display the original matrix and the predicted ratings (reconstructed matrix)
print("\nOriginal Matrix B:")
print(B)

print("\nReconstructed Matrix (Predicted Ratings):")
print(predicted_matrix)