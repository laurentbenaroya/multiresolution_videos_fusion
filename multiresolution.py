import cv2
import numpy as np

"""
Laplacian multi-resolution pyramid
"""


def build_gaussian_pyramid(src, level = 5):
    s = src.copy()
    pyramid = [s]
    for i in range(level):
        s = cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid


def build_laplacian_pyramid(src, levels = 5):
    gaussian_pyramid = build_gaussian_pyramid(src, levels)
    pyramid = [gaussian_pyramid[-1]]  # Add the smallest level of the Gaussian pyramid
    for i in range(levels, 0, -1):
        GE = cv2.pyrUp(gaussian_pyramid[i])
        L = cv2.subtract(gaussian_pyramid[i-1], GE)
        pyramid.append(L)

    return pyramid[::-1]


def reconstruct_from_laplacian_pyramid(lap_pyr):
    lap_pyr = lap_pyr[::-1]
    reconstructed_image = lap_pyr[0]
    for i in range(1, len(lap_pyr)):
        size = (lap_pyr[i].shape[1], lap_pyr[i].shape[0])
        reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
        reconstructed_image = cv2.add(reconstructed_image, lap_pyr[i])
    return reconstructed_image


if __name__ == '__main__':
    # Load an image
    image = cv2.imread('images/Attal/1.png')
    reconstructed_image = np.zeros_like(image)
    for c in range(3):
        image_c = image[:, :, c]
        image_c = image_c.astype(np.float32)
        image_c = image_c / 255.0

        # Build the Laplacian pyramid
        lap_pyr = build_laplacian_pyramid(image_c)

        # Reconstruct the image
        reconstructed_image_c = reconstruct_from_laplacian_pyramid(lap_pyr)
        reconstructed_image_c = reconstructed_image_c * 255
        reconstructed_image_c = reconstructed_image_c.astype(np.uint8)
        if c == 0:
            reconstructed_image = np.zeros_like(image)
        reconstructed_image[:, :, c] = reconstructed_image_c

    # Display the original and reconstructed images for comparison
    cv2.imshow('Original', image)
    cv2.imshow('Reconstructed', reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()