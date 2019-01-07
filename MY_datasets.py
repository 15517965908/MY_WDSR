import cv2
import random
import numpy as np
import os

scale = 4
lr_img_size = 16
hr_img_size = lr_img_size*scale
bath_size = 32


def read_img(lr_path, hr_path):
    while True:
        lr_img_list = os.listdir(lr_path)
        random.shuffle(lr_img_list)
        for lr_img_name in lr_img_list[0:32]:
            lr = cv2.imread(lr_path+lr_img_name)
            lr_shape = lr.shape
            hr = cv2.imread(hr_path+lr_img_name)

            ran = [random.randint(0, lr_shape[0]-lr_img_size-1), random.randint(0, lr_shape[1]-lr_img_size-1)]
            lr = lr[ran[0]:ran[0]+lr_img_size, ran[1]:ran[1]+lr_img_size, :]
            hr = hr[ran[0]*scale:ran[0]*scale+hr_img_size, ran[1]*scale:ran[1]*scale+hr_img_size, :]
            try:
                lr, hr = flip_img(lr, hr)
            except Exception:
                print('error')
                continue
            lr = np.expand_dims(lr, axis=0)
            hr = np.expand_dims(hr, axis=0)
            yield lr, hr


def read_img_bs(lr_path, hr_path):
    while True:
        lr_img_list = os.listdir(lr_path)
        random.shuffle(lr_img_list)
        for lr_img_name in lr_img_list[0:32]:
            lr = cv2.imread(lr_path+lr_img_name)
            lr_shape = lr.shape
            hr = cv2.imread(hr_path+lr_img_name)

            for _ in range(8):
                ran = [random.randint(0, lr_shape[0]-lr_img_size-1), random.randint(0, lr_shape[0]-lr_img_size-1)]
                lr = lr[ran[0]:ran[0]+lr_img_size, ran[1]:ran[1]+lr_img_size, :]
                hr = hr[ran[0]*scale:ran[0]*scale+hr_img_size, ran[1]*scale:ran[1]*scale+hr_img_size, :]
                try:
                    lr, hr = flip_img(lr, hr)
                except Exception:
                    print('error')
                    continue
                lr = np.expand_dims(lr, axis=0)
                hr = np.expand_dims(hr, axis=0)
                yield lr, hr


def flip_img(lr, hr):
    p = random.randint(-1, 1)
    lr = cv2.flip(lr, p, dst=None)
    hr = cv2.flip(hr, p, dst=None)
    return lr, hr


def test_read_img():
    lr_path_test = 'images/train/LR/'
    hr_path_test = 'images/train/HR/'
    x = read_img(lr_path_test, hr_path_test)
    for _ in range(1000):
        xlr, xhr = next(x)
        xlr = xlr[0, :, :, :].astype(np.uint8)
        xhr = xhr[0, :, :, :].astype(np.uint8)
        # xlr, xhr = flip_img(xlr, xhr)
        if xlr.shape != (lr_img_size, lr_img_size, 3):
            print('here error')
        print(xlr.shape, xhr.shape)
    cv2.imshow('lr', xlr)
    cv2.imshow('hr', xhr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # lr_path = 'images/train/LR/'
    # hr_path = 'images/train/HR/'
    test_read_img()



