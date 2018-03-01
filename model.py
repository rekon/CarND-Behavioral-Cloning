import os
import numpy as np
import pandas as pd
import random
import matplotlib.image as mpimage
from sklearn import model_selection

from keras import models, optimizers, backend
from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

ROOT_PATH = './'
BATCH_SIZE = 64
EPOCHS = 10

# Camera positions
CAMERAS = ['left', 'center', 'right']
CAMERA_STEERING_CORRECTION = [.25, 0., -.25]

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 320
IMAGE_CHANNEL = 3

#fuction to read image from file
def get_image(index, data, should_augment):
    #randomly select left, right or center camera position if should_augment = True
    camera = np.random.randint(len(CAMERAS)) if should_augment else 1
    # Read image and appropiately add offset to the steering angle
    image = mpimage.imread(os.path.join(
        ROOT_PATH, data[CAMERAS[camera]].values[index].strip()))
    angle = data.steering.values[index] + CAMERA_STEERING_CORRECTION[camera]

    return [image, angle]

#generator function to return images batchwise
def generator(data, should_augment=False):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])
        for batch in range(0, len(indices_arr), BATCH_SIZE):
            #slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + BATCH_SIZE)]

            #initializing the arrays, x_train and y_train
            x_train = np.empty(
                [0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
            y_train = y = np.empty([0], dtype=np.float32)

            for i in current_batch:
                #get an image and its corresponding steering angle for an image
                [image, angle] = get_image(i, data, should_augment)

                # Appending them to existing batch
                x_train = np.append(x_train, [image], axis=0)
                y_train = np.append(y_train, [angle])

            # Horizontally flip half of images in the batch
            flip_indices = random.sample(
                range(x_train.shape[0]), int(x_train.shape[0] / 2))
            x_train[flip_indices] = x_train[flip_indices, :, ::-1, :]
            #changing the sign of their steering angles accordingly
            y_train[flip_indices] = -y_train[flip_indices]

            yield (x_train, y_train)


def get_model(time_len=1):

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL),
                     output_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)))
    model.add(Cropping2D(cropping=((65, 25), (0, 0)),
                         input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    model.summary()

    return model


if __name__ == "__main__":

    data = pd.read_csv(os.path.join(ROOT_PATH, 'driving_log.csv'))

    # Split data into random training and validation sets
    d_train, d_valid = model_selection.train_test_split(data, test_size=.2)

    train_gen = generator(d_train, True)
    validation_gen = generator(d_valid, False)

    model = get_model()
    model.fit_generator(
        train_gen,
        samples_per_epoch=d_train.shape[0],
        nb_epoch=EPOCHS,
        validation_data=validation_gen,
        nb_val_samples=d_valid.shape[0],
        verbose=1
    )
    print("Saving model..")

    model.save("./model.h5")

    print("Model Saved successfully!!")

    #Destroying the current TF graph to avoid clutter from old models / layers
    backend.clear_session()
