import numpy as np
import matplotlib.pyplot as plt


class Regression:
    def __init__(self):
        pass

    @staticmethod
    def find_sum(l, p):
        res = 0
        for i in l:
            res += i ** p
        return res

    @staticmethod
    def find_mul_sum(l1, l2):
        res = 0
        for i in range(len(l1)):
            res += (l1[i] * l2[i])
        return res

    @staticmethod
    def solve_equ(x, sum_x, sum_x2, sum_y, sum_xy):
        # Equation no 1
        # Ey = a * Ex + b * n

        # Equation no 2
        # Exy = a * Ex^2 + b * Ex

        n = len(x)

        p = np.array([[sum_x2, sum_x], [sum_x, n]])
        q = np.array([sum_xy, sum_y])

        res = np.linalg.solve(p, q)
        return res

    @staticmethod
    def predict(x, res):
        y_pred = []
        for i in x:
            y_pred.append(res[0] * i + res[1])
        return y_pred


def main():
    # Input matrix
    A = np.array([
        [2, 1],
        [1, 0],
        [0, 1]
    ])

    # Extract x and y from the matrix
    x = A[:, 0]
    y = A[:, 1]

    r = Regression

    sum_x = r.find_sum(x, 1)
    sum_y = r.find_sum(y, 1)
    sum_x2 = r.find_sum(x, 2)
    sum_xy = r.find_mul_sum(x, y)

    res = r.solve_equ(x, sum_x, sum_x2, sum_y, sum_xy)

    # Calculate predictions
    y_pred = r.predict(x, res)

    # Print the line equation
    print(f"Line equation: y = {res[0]:.2f}x + {res[1]:.2f}")

    # Plot the points and the regression line
    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x, y_pred, color='blue', label='Regression line')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
