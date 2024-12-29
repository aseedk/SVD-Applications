import pprint

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from Calculate_SVD import calculate_svd

class Regression:
    def __init__(self):
        pass

    @staticmethod
    def compute_svd_solution(A, B):
        # Compute SVD of matrix A without libraries
        U, singular_values, Vt = calculate_svd(A)
        # Construct the diagonal matrix of singular values
        S = np.diag(singular_values)
        # Compute the pseudo-inverse of S
        S_pseudo_inv = np.linalg.pinv(S)

        # Calculate the pseudo-inverse of A using SVD components
        A_pseudo_inv = Vt.T @ S_pseudo_inv @ U.T

        # Solve the least-squares problem
        X = A_pseudo_inv @ B

        print("Least squares solution:")
        pprint.pprint(X)
        return X

    @staticmethod
    def predict(A, x):
        # Predict values using A and x
        return A @ x


def main():
    # Input matrices
    A = np.array([
        [2, 1],
        [1, 0],
        [0, 1]
    ], dtype=np.float64)
    # define a matrix
    # perform SVD

    B = np.array([
        [1],
        [2],
        [2]
    ])

    r = Regression()

    # Compute least squares solution using custom SVD
    x = r.compute_svd_solution(A, B)

    # Predict values
    y_pred = r.predict(A, x)

    # Print the results
    print(f"Predicted values (Ax)")
    pprint.pprint(y_pred)

    # Print the regression equation
    print(f"Regression Equation: y = {x[0][0]:.2f}x1 + {x[1][0]:.2f}")

if __name__ == "__main__":
    main()
