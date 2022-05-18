import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def gaussian_pyramid(image, n, filt_size=5, std=1):
    pyramid = [cv.GaussianBlur(image, (filt_size, filt_size), std, std)]
    for i in range(1, n):
        pyramid.append(cv.GaussianBlur(pyramid[i - 1], (filt_size, filt_size), std, std)[::2, ::2])
    return pyramid


def laplacian_pyramid(image, n, filt_size=5, std=1):
    gauss_pyramid = gaussian_pyramid(image, n, filt_size=filt_size, std=std)
    pyramid = []
    for i in range(1, n):
        size = (gauss_pyramid[i - 1].shape[1], gauss_pyramid[i - 1].shape[0])
        pyramid.append(-cv.resize(gauss_pyramid[i], size) + gauss_pyramid[i - 1])
    pyramid.append(gauss_pyramid[-1])
    return pyramid


def reconstruct(pyramid):
    size = (pyramid[0].shape[1], pyramid[0].shape[0])
    result = np.zeros((size[1], size[0]))
    for i in pyramid[::-1]:
        result += cv.resize(i, size)
    return result


def multires_blend(im1, im2, mask, n=10, filt_size=5, std=1):
    gauss_mask = gaussian_pyramid(mask, n, filt_size=filt_size, std=std)
    for i in range(len(gauss_mask)):
        plt.imsave(f'images/example1/progress/{i}.jpg', gauss_mask[i], cmap='gray')
    lapl_im1 = laplacian_pyramid(im1, n, filt_size=filt_size, std=std)
    lapl_im2 = laplacian_pyramid(im2, n, filt_size=filt_size, std=std)
    lapl_res = []
    for i in range(n):
        lapl_res.append(lapl_im2[i] * gauss_mask[i] + (1 - gauss_mask[i]) * lapl_im1[i])
    return reconstruct(lapl_res)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    source = cv.imread('images/example1/source.jpg', cv.IMREAD_GRAYSCALE)/255
    target = cv.imread('images/example1/target.jpg', cv.IMREAD_GRAYSCALE)/255
    mask = cv.imread('images/example1/mask.png', cv.IMREAD_GRAYSCALE)/255
    result = multires_blend(source, target, mask)
    plt.imsave('images/example1/result.jpg', result, cmap='gray')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
