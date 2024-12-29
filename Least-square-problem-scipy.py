import numpy as np
from scipy.linalg import svd

# Matrices A and B
A = np.array([
    [2, 1],
    [1, 0],
    [0, 1]
], dtype=np.float64)

B = np.array([
    [1],
    [2],
    [2]
], dtype=np.float64)

# Perform Singular Value Decomposition
U, s, Vt = svd(A, full_matrices=False)
# print different components
print("U: ", U)
print("Singular array", s)
print("V^{T}", Vt)
# Construct the diagonal matrix of singular values
S = np.diag(s)

# Compute the pseudo-inverse of S
S_pseudo_inv = np.linalg.pinv(S)

# Calculate the pseudo-inverse of A using SVD components
A_pseudo_inv = Vt.T @ S_pseudo_inv @ U.T

# Solve the least-squares problem
X = A_pseudo_inv @ B

print("Least squares solution:")
print(X)
