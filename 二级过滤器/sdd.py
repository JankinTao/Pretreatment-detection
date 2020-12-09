import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from cv2 import cv2
import os
import shutil
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
    start = timer()
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    end = timer()
    seconds = end - start
    timing.append(seconds)
    print("Time taken : {0} seconds".format(seconds))

    return np.mean(np.mean(ssim_map))


def read_img(directory_name):
    for filename in os.listdir(r'./' + directory_name):
        img_name.append(filename)


if __name__ == "__main__":
    img_name = []
    read_img('data')
    lenth = len(img_name)
    print(lenth)
    ssim_diff = []
    timing = []
    i = 0
    j = 0
    while True:
        print(j)
        if j>lenth-1:
            break
        else:
            im1 = cv2.imread("data" + "/" + img_name[i], cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread("data" + "/" + img_name[j], cv2.IMREAD_GRAYSCALE)
            a = im1.shape
            b = im2.shape
            im1 = cv2.resize(im1, (int(a[1] / 5), int(a[0] / 5)), interpolation=cv2.INTER_AREA)
            im2 = cv2.resize(im2, (int(a[1] / 5), int(a[0] / 5)), interpolation=cv2.INTER_AREA)
            if j==0:
                old_path = './data/'
                new_path = './handle/'
                full_file = os.path.join(old_path, img_name[j])
                new_full_file = os.path.join(new_path, img_name[j])
                shutil.copy(full_file, new_full_file)

            diff = compute_ssim(np.array(im1), np.array(im2))
            print(diff)
            ssim_diff.append(diff)

            if diff>("yuzhi"):
                j+=1
            else:
                old_path = './data/'
                new_path = './handle/'
                full_file = os.path.join(old_path, img_name[j])
                new_full_file = os.path.join(new_path, img_name[j])
                shutil.copy(full_file, new_full_file)
                i=j
                j+=1
    x = np.arange(0,lenth)
    plt.figure()
    plt.plot(x, ssim_diff, color="red", linewidth=2, linestyle='solid', marker='o')
    plt.xlabel('number of frames')
    plt.ylabel('difference')
    plt.title('Difference distribution')
    plt.savefig('./png/差异值分布.png')
    plt.figure()
    plt.bar(x,timing,fc = 'g')
    plt.xlabel('number of frames')
    plt.ylabel('time')
    plt.title('Time distribution')
    plt.savefig('./png/时间分布.png')
    plt.show()

