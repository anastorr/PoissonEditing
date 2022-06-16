import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def find_border(mask):
    border = cv.filter2D(mask, cv.CV_64F, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    border = np.where(border > 0, 1.0, 0)
    border = border-mask
    return border


def shift(f, n, axis):
    f = np.roll(f, n, axis=axis)
    if n == -1 and axis == 0:
        f[-1, :] = 0
    elif n == 1 and axis == 0:
        f[0, :] = 0
    elif n == -1 and axis == 1:
        f[:, -1] = 0
    elif n == 1 and axis == 1:
        f[:, 1] = 0
    return f


def compute_Np(shape):
    Np = np.full(shape, 4)
    Np[0, :] = 3
    Np[-1, :] = 3
    Np[:, 0] = 3
    Np[:, -1] = 3
    Np[[0, -1, 0, -1], [0, 0, -1, -1]] = 2
    return Np


def neighbours(mask):
    idx = np.nonzero(mask)
    mask_ind = np.zeros(mask.shape, dtype=int)
    mask_ind[idx] = np.arange(1, idx[0].size+1, 1)
    bottom = shift(mask_ind, -1, 0)[idx]
    top = shift(mask_ind, 1, 0)[idx]
    left = shift(mask_ind, 1, 1)[idx]
    right = shift(mask_ind, -1, 1)[idx]
    return top, bottom, left, right


def shift_ind(f):
    left_x, left_y = np.arange(0, f.shape[0], 1), np.append(np.arange(1, f.shape[1], 1), f.shape[1]-1)
    right_x, right_y = np.arange(0, f.shape[0], 1), np.insert(np.arange(0, f.shape[1]-1, 1), 0, 0)
    bottom_x, bottom_y = np.append(np.arange(1, f.shape[0], 1), f.shape[0]-1), np.arange(0, f.shape[1], 1)
    top_x, top_y = np.insert(np.arange(0, f.shape[0]-1, 1), 0, 0), np.arange(0, f.shape[1], 1)
    return (np.repeat(left_x, f.shape[1]), np.tile(left_y, f.shape[0]),
            np.repeat(right_x, f.shape[1]), np.tile(right_y, f.shape[0]),
            np.repeat(bottom_x, f.shape[1]), np.tile(bottom_y, f.shape[0]),
            np.repeat(top_x, f.shape[1]), np.tile(top_y, f.shape[0]))


def A(f, Np, ind):
    result = np.zeros(f.shape)
    result[:-1] = Np*f[:-1] - (f[ind[0]-1] + f[ind[1]-1] + f[ind[2]-1] + f[ind[3]-1])
    return result


def poisson_blending(source, target, mask, eps=0.001, channel=0):
    ind_mask = np.nonzero(mask)
    N = ind_mask[0].size
    f = np.zeros(N+1)
    f[:-1] = source[ind_mask]
    Np = compute_Np(mask.shape)[ind_mask]

    border_mask = find_border(mask)
    target_border_masked = target*border_mask

    left_x, left_y, right_x, right_y, bottom_x, bottom_y, top_x, top_y = shift_ind(mask)

    b1 = (target_border_masked[left_x, left_y] + target_border_masked[right_x, right_y] +
          target_border_masked[top_x, top_y] + target_border_masked[bottom_x, bottom_y]).reshape(mask.shape)[ind_mask]
    b2 = (source[left_x, left_y] + source[right_x, right_y] +
          source[top_x, top_y] + source[bottom_x, bottom_y]).reshape(mask.shape)[ind_mask]
    b = - b2 + b1 + Np*source[ind_mask]

    idx = neighbours(mask)
    delta = np.zeros(N+1)
    delta[:-1] = A(f, Np, idx)[:-1] - b
    i = 0
    while abs(delta).max() > eps:
        f = f - 0.1*delta
        print(f"channel{channel}, step{i}, del{abs(delta).max()}")
        delta[:-1] = A(f, Np, idx)[:-1] - b
        i += 1
    result = target
    result[ind_mask] = f[:-1]
    return result


def poisson_blending_mix(source, target, mask, n, eps=0.001, channel=0):
    ind_mask = np.nonzero(mask)
    N = ind_mask[0].size
    f = np.zeros(N+1)
    f[:-1] = source[ind_mask]
    Np = compute_Np(mask.shape)[ind_mask]

    border_mask = find_border(mask)
    target_border_masked = target*border_mask

    left_x, left_y, right_x, right_y, bottom_x, bottom_y, top_x, top_y = shift_ind(mask)

    b1 = (target_border_masked[left_x, left_y] + target_border_masked[right_x, right_y] +
          target_border_masked[top_x, top_y] + target_border_masked[bottom_x, bottom_y]).reshape(mask.shape)[ind_mask]
    b2 = (source[left_x, left_y] + source[right_x, right_y] +
          source[top_x, top_y] + source[bottom_x, bottom_y]).reshape(mask.shape)[ind_mask]
    b21 = (target[left_x, left_y] + target[right_x, right_y] +
           target[top_x, top_y] + target[bottom_x, bottom_y]).reshape(mask.shape)[ind_mask]

    mix = abs(Np*source[ind_mask]-b2) > abs(Np*target[ind_mask]-b21)
    mixing = np.zeros(mask.shape)
    mixing[ind_mask] = mix
    plt.imsave(f'images/example{n}/mixing.png', mixing, cmap='gray')
    b = np.where(mix, - b2 + b1 + Np*source[ind_mask], - b21 + b1 + Np*target[ind_mask])

    idx = neighbours(mask)
    delta = np.zeros(N+1)
    delta[:-1] = A(f, Np, idx)[:-1] - b
    i = 0
    while abs(delta).max() > eps:
        f = f - 0.1*delta
        print(f"channel{channel}, step{i}, del{abs(delta).max()}")
        delta[:-1] = A(f, Np, idx)[:-1] - b
        i += 1
    result = target
    result[ind_mask] = f[:-1]
    return result


def example(n, eps):
    mask = cv.imread(f'images/example{n}/mask.png')/255
    source = cv.imread(f'images/example{n}/source.jpg')/255
    target = cv.imread(f'images/example{n}/target.jpg')/255
    result1 = poisson_blending(source[..., 0], target[..., 0], mask[..., 0], eps=eps, channel=1)
    result2 = poisson_blending(source[..., 1], target[..., 1], mask[..., 1], eps=eps, channel=2)
    result3 = poisson_blending(source[..., 2], target[..., 2], mask[..., 2], eps=eps, channel=3)

    result = np.array([result3.T, result2.T, result1.T]).T
    plt.imsave(f'images/example{n}/result_mix2.jpg', np.clip(result, 0, 1))


example(12, 0.0001)
