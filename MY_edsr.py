from model.edsr import edsr
from callback import learning_rate, model_checkpoint_after, tensor_board
from keras.losses import mean_absolute_error, mean_squared_error
from train import psnr

import cv2
import keras
from keras.optimizers import Adam
from MY_datasets import read_img


scale = 4
loss = mean_absolute_error
weight = 'weight/sr.h5'
logs = 'logs/'
learning_rate_step_size = 1e-4
learning_rate_decay = 200
lr_path = 'images/train/LR/'
hr_path = 'images/train/HR/'
lr_validation = 'images/test/LR/'
hr_validation = 'images/test/HR/'


def main():
    sr = edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None, tanh_activation=True)
    sr.compile(optimizer=Adam(lr=1e-4), loss=loss, metrics=[psnr])
    sr.summary()
    # sr.save_weights(weight)
    sr.load_weights(weight)

    callbacks = [
            tensor_board(logs),
            learning_rate(step_size=learning_rate_step_size, decay=learning_rate_decay)
            # model_checkpoint_after(save_models_after_epoch, models_dir, monitor=f'val_psnr',
            #                        save_best_only=save_best_models_only or benchmark)
        ]

    sr.fit_generator(read_img(lr_path, hr_path),
                     epochs=10000, steps_per_epoch=8,
                     validation_data=read_img(lr_validation, hr_validation),
                     validation_steps=5,
                     callbacks=callbacks)
    sr.save_weights(weight)


if __name__ == '__main__':
    main()
    pass
