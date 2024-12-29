import pprint

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from Calculate_SVD import calculate_svd

# Use the provided matrix as the dataset
matrix = np.array([
    [2, 1],
    [1, 0],
    [0, 1]
], dtype=np.float64)

# Generate labels for the matrix (example: assign labels for each row)
labels = np.array([1, 1, 1])  # Example: Assigning arbitrary labels (modify as needed)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(matrix)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=1, random_state=42)

# Compute mean images and SVD for each label class
mean_images = {}
svd_components = {}

for label in np.unique(y_train):
    label_images = X_train[y_train == label]
    mean_image = np.mean(label_images, axis=0)
    U, s, Vh = calculate_svd(mean_image.reshape(-1, 1))  # Reshaped for SVD computation
    print("\nU Matrix: ")
    pprint.pprint(U)
    print("\ns Matrix: ")
    pprint.pprint(s)
    print("\nVh Matrix: ")
    pprint.pprint(Vh)
    mean_images[label] = mean_image
    svd_components[label] = (U, s, Vh)

# Function to classify a new data point
def classify_point(point):
    min_error = float('inf')
    predicted_label = None
    for label_inner in np.unique(y_train):
        U_inner, s_inner, Vh_inner = svd_components[label_inner]
        print("\nU Matrix: ")
        pprint.pprint(U_inner)
        print("\ns Matrix: ")
        pprint.pprint(s_inner)
        print("\nVh Matrix: ")
        pprint.pprint(Vh_inner)
        # Project the point onto the subspace
        projection = U_inner @ np.diag(s_inner) @ Vh_inner
        # Compute reconstruction error
        error = np.linalg.norm(point.reshape(-1, 1) - projection)
        if error < min_error:
            min_error = error
            predicted_label = label_inner
    return predicted_label

# Predict on the test set
y_pred = [classify_point(img) for img in X_test]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Classification accuracy: {accuracy * 100:.2f}%')
