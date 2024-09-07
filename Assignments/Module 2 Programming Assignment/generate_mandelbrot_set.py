"""
Starter code for the fast generation of the Mandelbrot Set.
"""

import time

import numpy as np
from numba import jit
from pylab import imshow, show


@jit(nopython=True)
def mandel(x, y, max_iters):
    """
    Computes the behavior of '0' under max_iters iterations for the value c.
    """
    c = complex(
        x, y
    )  # Generate complex number, c, given real and imaginary components.
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c  # Compute the function.
        if (
            z.real * z.real + z.imag * z.imag
        ) >= 4:  # Return iteration value if z becomes larger than 4.
            return i

    return max_iters  # Return max_iters otherwise.


@jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, img, iters):
    """
    The Mandelbrot Set is a fractal. Create the fractal.
    """
    height = img.shape[0]
    width = img.shape[1]

    # Calculate pixel sizes.
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            img[y, x] = color # Assign a color to the image.


if __name__ == "__main__":
    image = np.zeros((1024, 2024), dtype=np.uint8)  # Generate the image.
    start = time.time()  # Get the start time.
    create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
    end = time.time()  # Get the end time.
    print(f"Elapsed = {(end - start)}")  # Print the elapsed time.
    imshow(image)
    show() # Show the Mandelbrot Set.
