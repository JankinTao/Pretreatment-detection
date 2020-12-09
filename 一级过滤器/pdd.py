import cv2
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
# from pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']
 
def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n
 
 
def dHash(img):
    img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
 
    return hash_str
 
 
 
def get_frame(file_path):

    cap = cv2.VideoCapture(file_path)
    dict_img = {"1": [], }
    img_frame = ("jiangezhen")
    img_dif = ("chayizhi")
    diff = []
    timing = []

    if cap.isOpened():
        rate = int(cap.get(5))
        FrameNumber = cap.get(7)
        timeF = int(FrameNumber // img_frame)
        print(rate)
        print(timeF)
        print(FrameNumber)
        c = 0

        while True:
            success, frame = cap.read()
            if c==0:
                cv2.imwrite('img/{}.jpg'.format(c), frame)
            if c > FrameNumber:
                print('******************')
                break
            else:
                if "2" in dict_img:
                    dict_img["1"] = dict_img["2"]
                else:
                    dict_img["1"] =dHash(frame)
                dict_img["2"] = dHash(frame)
                start = timer()
                sm_img = cmpHash(dict_img["1"], dict_img["2"])
                end = timer()
                seconds = end - start
                timing.append(seconds)
                print("Time taken : {0} seconds".format(seconds))
                print(sm_img)
                diff.append(sm_img)



            if sm_img:
                if sm_img >=img_dif:
                    cv2.imwrite('img/{}.jpg'.format(c), frame)
                    print(c)
            c += img_frame
            if not success:
                print('video is all read')
                break
        i = np.arange(0, FrameNumber, 5)
        plt.figure()
        plt.plot(i, diff, color="red", linewidth=2, linestyle='-', marker='+')
        plt.xlabel('number of frames')
        plt.ylabel('difference')
        plt.title('Difference distribution')
        plt.savefig('./差异值分布1.png')

        plt.figure()
        plt.bar(i, timing, fc='g')
        plt.xlabel('number of frames')
        plt.ylabel('time')
        #plt.yscale('log')
        plt.title('Time distribution')
        plt.savefig('./时间分布1.png')
        plt.show()
        print(len(timing))
        print(len(diff))

if __name__ == '__main__':

    get_frame('./data/MP4')




