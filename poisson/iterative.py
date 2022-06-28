import numpy as np
from numba import njit


@njit
def iter_A(f, Np, ind):
    result = np.zeros(f.shape)
    for i in range(f.shape[0]-1):
        result[i] = Np[i]*f[i] - (f[ind[0][i] - 1] + f[ind[1][i] - 1] + f[ind[2][i] - 1] + f[ind[3][i] - 1])
    return result


def A(f, Np, ind):
    result = np.zeros(f.shape)
    result[:-1] = Np*f[:-1] - (f[ind[0] - 1] + f[ind[1] - 1] + f[ind[2] - 1] + f[ind[3] - 1])
    return result


def fixed_point(f, b, delta, eps, Np, idx, gamma, channel):
    i = 0
    while abs(delta).max() > eps:
        f = f - gamma*delta
        print(f"channel{channel}, step{i}, del{abs(delta).max()}")
        delta[:-1] = iter_A(f, Np, idx)[:-1] - b
        # if i%save_step == 0 and save_progress is True:
        #     target[ind_mask] = f[:-1]
        #     plt.imsave(f'images/example{n}/progress/chan{channel}step{i:06d}gamma{gamma}.jpg', np.clip(target, 0, 1), cmap='gray')
        i += 1
    return f


def gradient_descent(f, b, delta, eps, Np, idx, gamma, channel):
    i = 0
    while abs(delta).max() > eps:
        f = f - gamma*iter_A(delta, Np, idx)
        print(f"channel{channel}, step{i}, del{abs(delta).max()}")
        delta[:-1] = iter_A(f, Np, idx)[:-1] - b
        i += 1
    return f


@njit
def iter_gauss_seidel(f, b, idx, Np):
    f_next = np.zeros(f.shape)
    for i in range(f.shape[0] - 1):
        f_next[i] = (b[i] + f_next[idx[0][i] - 1] + f_next[idx[1][i] - 1] + f[idx[2][i] - 1] + f[idx[3][i] - 1]) / Np[i]
    # delta = f_next - f
    return f_next


def gauss_seidel(f, b, delta, eps, Np, idx, channel):
    j = 0
    while abs(delta).max() > eps:
        print(f"channel{channel}, step{j}, del{abs(delta).max()}")
        f = iter_gauss_seidel(f, b, idx, Np)
        delta[:-1] = iter_A(f, Np, idx)[:-1] - b
        j += 1
    return f