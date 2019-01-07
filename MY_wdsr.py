from model.wdsr import wdsr_b
from callback import learning_rate, model_checkpoint_after, tensor_board
from keras.losses import mean_absolute_error, mean_squared_error
from train import psnr as psnr_tf

# import tensorflow as tf
import cv2
import os
import numpy as np
import keras
from keras.optimizers import Adam
from MY_datasets import read_img
from keras.callbacks import ModelCheckpoint


scale = 4
loss = mean_squared_error
weight = 'weight/wdsr.h5'
logs = 'logs/wdsr/'
learning_rate_step_size = 1e-4
learning_rate_decay = 200
lr_path = 'images/train/LR/'
hr_path = 'images/train/HR/'
lr_validation = 'images/test/LR/'
hr_validation = 'images/test/HR/'


def main():
    sr = wdsr_b(scale)
    sr.compile(optimizer=Adam(lr=1e-4), loss=loss, metrics=[psnr_tf])
    sr.summary()
    # sr.save_weights(weight)
    sr.load_weights(weight)

    callbacks = [
            tensor_board(logs),
            learning_rate(step_size=learning_rate_step_size, decay=learning_rate_decay),
            ModelCheckpoint(weight, monitor='val_loss', verbose=1,
                            save_best_only=True, save_weights_only=True,)
            # model_checkpoint_after(save_models_after_epoch, models_dir, monitor=f'val_psnr',
            #                        save_best_only=save_best_models_only or benchmark)
        ]

    sr.fit_generator(read_img(lr_path, hr_path),
                     epochs=10000, steps_per_epoch=32,
                     validation_data=read_img(lr_validation, hr_validation),
                     validation_steps=5,
                     callbacks=callbacks)
    # sr.save_weights(weight)


def psnr(hr, sr):
    return 10 * np.log(255 * 2 / (np.mean(np.square(hr - sr))))


def test_model(hr, lr, bic, model):
    # main()

    sr = model.predict(np.expand_dims(lr, axis=0))
    sr = sr[0, :, :, :].astype(np.uint8)
    lr_hr = psnr(hr, sr)
    bic_sr = psnr(hr, bic)

    print('psnr_sr:', lr_hr, 'psnr_bic:', bic_sr)
    # cv2.imshow('sr', sr)
    # cv2.imshow('hr', hr)
    # cv2.imshow('lr', lr)
    # cv2.imshow('bic', bic)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def show_psnr(img_name, model):
    lr_img_path = 'images/test/LR/'
    hr_img_path = 'images/test/HR/'
    img_lr_path = lr_img_path + img_name
    img_hr_path = hr_img_path + img_name
    lr = cv2.imread(img_lr_path)
    bic = cv2.resize(lr, (lr.shape[1]*scale, lr.shape[0]*scale))
    hr = cv2.imread(img_hr_path)
    test_model(hr, lr, bic, model)


def use_it_test_model():
    model = wdsr_b(scale)
    model.load_weights(weight)
    for name in os.listdir(lr_validation):
        print(name)
        show_psnr(name, model)


if __name__ == '__main__':
    main()
