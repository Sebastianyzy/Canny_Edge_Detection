import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage


def get_norm(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power((x - mu) / sigma, 2) / 2)


# return the gaussian kernel for image smoothing
def gaussian_kernel(size, sigma):
    # compute 1D kernel
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = get_norm(kernel_1D[i], 0, sigma)
    # compute 2D kernel
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    # normalize
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D


# gaussian filter is used to smoothing the image
def gaussian_blur(img, sigma=None, kernel_size=None):
    kernel = gaussian_kernel(kernel_size, sigma)
    # convolve
    img = signal.convolve2d(img, kernel, mode='same')

    return img


# return the magnitude and the slope of the gradient
def sobel_filters(img, gx, gy):
    Ix = ndimage.filters.convolve(img, gx)
    Iy = ndimage.filters.convolve(img, gy)

    # gradient intensity and edge direction
    mag = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)

    return mag, theta


def canny_edge_detection(img, low=None, high=None, sigma=None, kernel_size=None):
    # convert to grayscale
    global neighb_1_x, neighb_1_y, neighb_2_y, neighb_2_x
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # noise reduction
    img = gaussian_blur(img, sigma, kernel_size)

    # get gradient
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # get magnitude ang theta for non-max suppression
    mag, ang = sobel_filters(img, gx, gy)

    mag_max = np.max(mag)
    if not low:
        low = mag_max * 0.1
    if not high:
        high = mag_max * 0.5

    # get height and width of the image
    height, width = img.shape

    # loop through all the pixels and find the maximum value in the edge directions.
    for i_x in range(width):
        for i_y in range(height):
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # x axis direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # top right
            elif 22.5 < grad_ang <= 67.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # y-axis
            elif 67.5 < grad_ang <= 112.5:
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            # top left
            elif 112.5 < grad_ang <= 157.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

            elif 157.5 < grad_ang <= 202.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue

            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    np.zeros_like(img)
    np.zeros_like(img)
    ids = np.zeros_like(img)

    # double threshold
    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < low:
                mag[i_y, i_x] = 0
            elif high > grad_mag >= low:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2
    return mag


def main():
    input_name = input("Select image:")
    img = input_name
    print("Apply canny_edge_detection on", img, "...")
    image = cv2.imread(img)
    canny_img = canny_edge_detection(image, low=10, high=255, sigma=3, kernel_size=5)
    plt.figure()
    plt.imshow(canny_img)
    print("Saving output image...")
    name = input("Enter image name:")
    plt.savefig(str(name))
    print("Image Saved.")
    plt.show()
    print("Edge Detection Complete.")
    print("Process complete.")


if __name__ == '__main__':
    main()