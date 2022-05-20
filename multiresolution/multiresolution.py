import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def clear_dir(path):
    if os.path.exists(path):
        for file_name in os.listdir(path):
            file = path + file_name
            if os.path.isfile(file):
                os.remove(file)
    else:
        os.makedirs(path)


def gaussian_pyramid(image, n, num, is_mask, filt_size=5, std=1):
    pyramid = [image]
    for i in range(1, n):
        size = (round(pyramid[i - 1].shape[1]*0.5), round(pyramid[i - 1].shape[0]*0.5))
        pyramid.append(cv.resize(cv.GaussianBlur(pyramid[i - 1], (filt_size, filt_size), std, std), size))
        if is_mask:
            plt.imsave(f'images/example{num}/gaussian pyramid/{i}.jpg',
                       (pyramid[i - 1] - pyramid[i - 1].min())/(pyramid[i - 1].max() - pyramid[i - 1].min()), cmap='gray')
    return pyramid


def laplacian_pyramid(image, n, num, filt_size=5, std=1):
    gauss_pyramid = gaussian_pyramid(image, n, num, False, filt_size=filt_size, std=std)
    pyramid = [gauss_pyramid[-1]]
    for i in range(1, n):
        pyramid.append(-cv.GaussianBlur(gauss_pyramid[-i-1], (filt_size, filt_size), std, std) + gauss_pyramid[-i-1])
        plt.imsave(f'images/example{num}/laplacian pyramid/{n-i}.jpg',
                   np.flip((pyramid[i] - pyramid[i].min())/(pyramid[i].max() - pyramid[i].min()),
                           axis=2))
    return list(reversed(pyramid))


# sum up all levels of a pyramid
def reconstruct(pyramid, num):
    size = (pyramid[0].shape[1], pyramid[0].shape[0])
    result = np.zeros(pyramid[0].shape)
    for i in range(len(pyramid)):
        result += cv.resize(pyramid[-i - 1], size)
        plt.imsave(f'images/example{num}/progress/{i}.jpg',
                   np.flip((result - result.min())/(result.max() - result.min()), axis=2))
    return result


def multires_blend(target, source, mask, num, use_grad_mask, mask_filt_size, n=10, filt_size=5, std=1):
    # compute gaussian pyramid for the mask
    gauss_mask = gaussian_pyramid(mask, n, num, True, filt_size=filt_size, std=std)
    # compute laplacian pyramids for images
    lapl_source = laplacian_pyramid(source, n, num, filt_size=filt_size, std=std)
    lapl_target = laplacian_pyramid(target, n, num, filt_size=filt_size, std=std)
    # compute laplacian pyramid of the resulting image
    lapl_res = []
    for i in range(n):
        coeff = (gauss_mask[i] if len(target.shape) == 2 else gauss_mask[i][..., np.newaxis])
        lapl_res.append(lapl_source[i]*coeff + (1 - coeff)*lapl_target[i])
    return reconstruct(lapl_res, num)


def grad_mask(mask, filt_size):
    std = (filt_size-1)/3
    return cv.GaussianBlur(mask, (filt_size, filt_size), std, std)


def example(num, method, use_grad_mask=True, mask_filt_size=5, **kwargs):
    clear_dir(f'images/example{num}/progress/')
    clear_dir(f'images/example{num}/gaussian pyramid/')
    clear_dir(f'images/example{num}/laplacian pyramid/')
    source = cv.imread(f'images/example{num}/source.jpg')/255
    target = cv.imread(f'images/example{num}/target.jpg')/255
    # mask = cv.imread(f'images/example{num}/mask.png', cv.IMREAD_GRAYSCALE)/255
    if use_grad_mask:
        mask = grad_mask(cv.imread(f'images/example{num}/mask.png', cv.IMREAD_GRAYSCALE), mask_filt_size)/255
    else:
        mask = cv.imread(f'images/example{num}/mask.png', cv.IMREAD_GRAYSCALE)/255
    result = method(target, source, mask, num, use_grad_mask, mask_filt_size, **kwargs)
    plt.imsave(f'images/example{num}/result.jpg', np.flip((result - result.min())/(result.max() - result.min()), axis=2))


if __name__ == '__main__':
    example(4, multires_blend, mask_filt_size=15, n=9, filt_size=5, std=1)

# TODO: apply gaussian filter or rescale in laplacian pyramid?
