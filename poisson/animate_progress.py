import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def animate_poisson(path):
    mask = cv.imread(path+'mask.png')[0]/255
    target = cv.imread(path+'mask.png')[2]/255
    n = len(os.listdir(path+'progress/'))
    ind_mask = np.nonzero(mask)
    for i in range(3):
        for file_name in os.listdir(path):
            file = path + file_name


