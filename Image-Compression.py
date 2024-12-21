import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from skimage.transform import resize

# Function to compress an image using SVD
def compress_image(filename):
    # Load the image and convert to grayscale
    image = color.rgb2gray(io.imread(filename))
    image = resize(image, (image.shape[0] // 2, image.shape[1] // 2), anti_aliasing=True)

    # Display the original image
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Perform SVD
    U, S, Vt = np.linalg.svd(image, full_matrices=False)

    # Define the ranks for compression
    ranks = [320, 160, 80, 40, 20, 10, 5]

    for i, rank in enumerate(ranks):
        # Compress by keeping only the top `rank` singular values
        compressed_S = np.zeros_like(S)
        compressed_S[:rank] = S[:rank]
        compressed_image = np.dot(U * compressed_S, Vt)

        # Display the compressed image
        plt.subplot(4, 2, i + 2)
        plt.imshow(compressed_image, cmap='gray')
        plt.title(f'Rank {rank} Image')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
compress_image('test_data/Image-Compression.jpg')
