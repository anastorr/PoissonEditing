import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def neigbours(arr, n):
    arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
    arr_top = arr[:-2, 1:-1].reshape(n)
    arr_bottom = arr[2:, 1:-1].reshape(n)
    arr_left = arr[1:-1, :-2].reshape(n)
    arr_right = arr[1:-1, 2:].reshape(n)
    return arr_top + arr_bottom + arr_left + arr_right


def product1_optimized(f, mask, border, source):
    n = f.size
    result = np.zeros(n)
    region_neighbours = neigbours(f*mask, n)
    border_neighbours = neigbours(border, n)
    source_neighbours = neigbours(source, n)
    f = f-source
    f[[0, -1], 1:-1] *= 3
    f[[0, 0, -1, -1], [0, -1, 0, -1]] *= 2
    f[1:-1, 1:-1] *= 4
    result = f.reshape(n) - region_neighbours - border_neighbours + source_neighbours
    return result


def product2_optimized(f, mask):
    n = f.size
    result = np.zeros(n)
    region_neighbours = neigbours(f*mask, n)
    f[[0, -1], 1:-1] *= 3
    f[[0, 0, -1, -1], [0, -1, 0, -1]] *= 2
    f[1:-1, 1:-1] *= 4
    result = f.reshape(n) - region_neighbours
    return result


def find_border(mask):
    border = cv.filter2D(mask, cv.CV_64F, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    border = np.where(border > 0, 1.0, 0)
    border = border-mask
    return border


def poisson_seamless_cloning(source, target, mask, eps=0.1, method=gradient_descent):
    border_mask = find_border(mask)
    border = target*border_mask
    f = np.zeros(mask.shape)
    delta = product1_optimized(f, mask, border, source)
    i = 0
    while np.sqrt((delta**2).sum()) > eps:
        f = f - 0.02*product2_optimized(delta.reshape(f.shape), mask).reshape(f.shape)
        i += 1
        print(f"step{i}, del{np.sqrt((delta**2).sum())}")
        if np.sqrt((delta**2).sum()) < 1.5:
            plt.imshow(f, cmap='gray')
            plt.show()
        delta = product1_optimized(f, mask, border, source)
    return f


mask = cv.imread('images/example5/mask.png', cv.IMREAD_GRAYSCALE)/255
source = cv.imread('images/example5/source.jpg', cv.IMREAD_GRAYSCALE)/255
target = cv.imread('images/example5/target.jpg', cv.IMREAD_GRAYSCALE)/255
result = poisson_seamless_cloning(source, target, mask)
plt.imshow(result, cmap='gray')
plt.show()