import numpy as np
from Calculate_SVD import calculate_svd

# Input matrices
A = np.array([
    [2, 1],
    [1, 0],
    [0, 1]
], dtype=np.float64)

B = np.array([
    [2],
    [1],
    [0]
], dtype=np.float64)

# Step 1: Compute SVD of matrix A
U, s, Vh = calculate_svd(A)

# Step 2: Compute the mean face (mean of columns of A)
mean_face = np.mean(A, axis=1).reshape(-1, 1)  # Column vector

# Step 3: Compute b_norm by subtracting the mean face from B
b_norm = B - mean_face

# Step 4: Compute b_recons
# Using the formula: b_recons = (b_norm^T · u1) u1 + (b_norm^T · u2) u2
u1 = U[:, 0].reshape(-1, 1)  # First left singular vector
u2 = U[:, 1].reshape(-1, 1)  # Second left singular vector

b_recons = (b_norm.T @ u1) * u1 + (b_norm.T @ u2) * u2

# Step 5: Compute b_final (reconstruct by adding back the mean face)
b_final = b_recons + mean_face

# Output results
print("Original B (vectorized face):")
print(B)

print("\nMean Face:")
print(mean_face)

print("\nNormalized B (B_norm):")
print(b_norm)

print("\nReconstructed B (B_recons):")
print(b_recons)

print("\nFinal Reconstructed Face (B_final):")
print(b_final)
