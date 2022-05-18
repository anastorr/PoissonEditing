import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def gaussian_pyramid(image, n, filt_size=5, std=1):
    pyramid = [image]
    for i in range(1, n):
        size = (round(pyramid[i - 1].shape[1]*0.5), round(pyramid[i - 1].shape[0]*0.5))
        pyramid.append(cv.resize(cv.GaussianBlur(pyramid[i - 1], (filt_size, filt_size), std, std), size))
    return pyramid


def laplacian_pyramid(image, n, filt_size=5, std=1):
    gauss_pyramid = gaussian_pyramid(image, n, filt_size=filt_size, std=std)
    pyramid = []
    for i in range(1, n):
        size = (gauss_pyramid[i - 1].shape[1], gauss_pyramid[i - 1].shape[0])
        pyramid.append(-cv.resize(gauss_pyramid[i], size) + gauss_pyramid[i - 1])
        plt.imsave(f'images/example1/laplacian pyramid/{i}.jpg',
                   np.flip((pyramid[i - 1] - pyramid[i - 1].min())/(pyramid[i - 1].max() - pyramid[i - 1].min()),
                           axis=2))
    pyramid.append(gauss_pyramid[-1])
    return pyramid


# sum up all levels of a pyramid
def reconstruct(pyramid):
    size = (pyramid[0].shape[1], pyramid[0].shape[0])
    result = np.zeros(pyramid[0].shape)
    for i in range(len(pyramid)):
        result += cv.resize(pyramid[-i - 1], size)
        plt.imsave(f'images/example1/progress/{i}.jpg',
                   np.flip((result - result.min())/(result.max() - result.min()), axis=2))
    return result


def multires_blend(im1, im2, mask, n=10, filt_size=5, std=1):
    # compute gaussian pyramid for the mask
    gauss_mask = gaussian_pyramid(mask, n, filt_size=filt_size, std=std)
    # compute laplacian pyramids for images
    lapl_im1 = laplacian_pyramid(im1, n, filt_size=filt_size, std=std)
    lapl_im2 = laplacian_pyramid(im2, n, filt_size=filt_size, std=std)
    # compute laplacian pyramid of the resulting image
    lapl_res = []
    for i in range(n):
        coeff = (gauss_mask[i] if len(im1.shape) == 2 else gauss_mask[i][..., np.newaxis])
        lapl_res.append(lapl_im2[i]*coeff + (1 - coeff)*lapl_im1[i])
    return reconstruct(lapl_res)


if __name__ == '__main__':
    source = cv.imread('images/example1/source.jpg')/255
    target = cv.imread('images/example1/target.jpg')/255
    mask = cv.imread('images/example1/mask.png', cv.IMREAD_GRAYSCALE)/255
    result = multires_blend(target, source, mask, n=10, filt_size=5, std=1)
    plt.imsave('images/example1/result.jpg', np.flip((result - result.min())/(result.max() - result.min()), axis=2))
