import numpy as np
from skimage import color, io
from skimage.transform import resize

import matplotlib.pyplot as plt

# Function to compute SVD without libraries
def compute_svd(matrix):
    """Compute SVD of a matrix without using libraries."""
    # Compute eigenvalues and eigenvectors
    AAT = np.dot(matrix, matrix.T)  # Compute A * A^T
    ATA = np.dot(matrix.T, matrix)  # Compute A^T * A

    U_eigenvalues, U = np.linalg.eigh(AAT)  # Compute eigenvectors for AAT
    V_eigenvalues, V = np.linalg.eigh(ATA)  # Compute eigenvectors for ATA

    # Sort eigenvalues and eigenvectors in descending order
    U_indices = np.argsort(U_eigenvalues)[::-1]
    V_indices = np.argsort(V_eigenvalues)[::-1]

    U = U[:, U_indices]
    V = V[:, V_indices]

    # Compute singular values
    singular_values = np.sqrt(np.maximum(U_eigenvalues[U_indices], 0))  # Ensure no negative values

    return U, singular_values, V.T


# Function to compress a matrix without library-based SVD
def compress_matrix_no_library(matrix):
    # Display the original matrix
    print("Original Matrix:")
    print(matrix)

    # Display the original image
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 2, 1)
    plt.imshow(matrix, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Perform custom SVD
    U, S, Vt = compute_svd(matrix)

    # Define the ranks for compression
    ranks = [2, 1]  # Since it's a 3x2 matrix, we will consider ranks 2 and 1

    for i, rank in enumerate(ranks):
        # Compress by keeping only the top `rank` singular values
        S_rank = np.zeros((rank, rank))
        np.fill_diagonal(S_rank, S[:rank])

        # Ensure correct shapes for matrix multiplication
        compressed_matrix = np.dot(np.dot(U[:, :rank], S_rank), Vt[:rank, :])

        # Display the compressed image
        plt.subplot(4, 2, i + 2)
        plt.imshow(compressed_matrix, cmap='gray')
        plt.title(f'Rank {rank} Image')
        plt.axis('off')
        # Output the compressed matrix after each rank
        print(f"\nMatrix after Rank {rank} compression:")
        print(compressed_matrix)

    plt.tight_layout()
    plt.show()


# Example usage

# Example usage
# Load the image and convert to grayscale
#matrix = color.rgb2gray(io.imread('test_data/Image-Compression.jpg'))
#matrix = resize(matrix, (matrix.shape[0] // 2, matrix.shape[1] // 2), anti_aliasing=True)
matrix = np.array([[2, 1], [1, 0], [0, 1]])  # Your given matrix
compress_matrix_no_library(matrix)

