"""
Reads in a noisy image, applies a Gaussian blurring filter to it, and saves the resulting image.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from PIL import Image

KERNEL_LENGTH = 5
SIGMA = 1


@jit(nopython=True)
def generate_kernel():
    """
    Generates Gaussian kernel with length and sigma.
    """
    ax = np.linspace(
        -(KERNEL_LENGTH - 1) / 2.0, (KERNEL_LENGTH - 1) / 2.0, KERNEL_LENGTH
    )
    gauss = np.exp(-0.5 * np.square(ax) / np.square(SIGMA))
    k = np.outer(gauss, gauss)
    return k / np.sum(k)  # Return the normalized kernel.


@jit(nopython=True)
def blurfilter(in_img, out_img, k):
    """
    Applies the Gaussian kernel to the noisy image and saves the resulting image.
    """
    for c in range(3):
        for x in range(in_img.shape[1]):
            for y in range(in_img.shape[0]):
                val = 0
                for i in range(-(KERNEL_LENGTH // 2), ((KERNEL_LENGTH // 2) + 1)):
                    for j in range(-(KERNEL_LENGTH // 2), ((KERNEL_LENGTH // 2) + 1)):
                        if (
                            (x + i < in_img.shape[1])
                            and (x + i >= 0)
                            and (y + j < in_img.shape[0])
                            and (y + j >= 0)
                        ):
                            val += int(in_img[y + j, x + i, c]) * k[i, j]
                out_img[y, x, c] = val


if __name__ == "__main__":
    img = np.array(Image.open("noisy1.jpg"))  # Open the noisy image.
    imgblur = img.copy()  # Save a copy of the noisy image.
    start = time.time()  # Get the start time.
    kernel = generate_kernel()  # Generate the kernel.
    blurfilter(img, imgblur, kernel)  # Blur the image.
    end = time.time()  # Get the end time.
    print(f"Elapsed = {(end - start)}")  # Print the elapsed time.

    # Display and save blurred image.
    fig = plt.figure()
    axes = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img)
    axes.set_title("Before")
    axes = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(imgblur)
    axes.set_title("After")
    img2 = Image.fromarray(imgblur)
    img2.save("blurred.jpg")
