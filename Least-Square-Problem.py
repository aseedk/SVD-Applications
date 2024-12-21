import numpy as np

def least_squares(A, b):
    """
    Solves the least squares problem: min ||A*x - b||^2

    Parameters:
    A (numpy.ndarray): The m x n matrix.
    b (numpy.ndarray): The m-dimensional vector.

    Returns:
    x (numpy.ndarray): The n-dimensional least squares solution vector.
    """
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Compute the pseudoinverse of A
    S_plus = np.diag(1 / S)  # Inverse of singular values
    A_plus = Vt.T @ S_plus @ U.T

    # Compute the least squares solution
    x = A_plus @ b
    return x

# Example usage
def main():
    # Define data points from the table
    x = np.array([1, 2, 3, 4, 5])  # x-coordinates
    y = np.array([2, 5, 3, 8, 7])  # y-coordinates

    # Construct the matrix A and vector b
    A = np.vstack([x, np.ones(len(x))]).T  # A = [[x1, 1], [x2, 1], ...]
    b = y  # b = [y1, y2, ...]

    # Solve the least squares problem to find [a, b] where y = ax + b
    solution = least_squares(A, b)

    print("Least squares solution (a, b):", solution)
    print(f"Line equation: y = {solution[0]:.2f}x + {solution[1]:.2f}")

if __name__ == "__main__":
    main()
