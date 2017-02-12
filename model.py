import argparse
import os, csv, re
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis
from keras import backend as K
import random
import cv2


def get_model(shape, load=False, checkpoint=None):
    """
    Return the model to be trained.
    Here, Nvidia's End-to-End deep learning architecture is used.
    """
    if load and checkpoint: return load_model(checkpoint)

    conv_layers1, conv_layers2 = [24, 36, 48], [64, 64]
    dense_layers = [100, 50, 10]

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape)) # normalize image
    model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # crop top 70 and bottom 25 pixel from image
    for cl in conv_layers1:
        model.add(Convolution2D(cl, 5, 5, subsample=(2, 2), activation='relu'))
    for cl in conv_layers2:
        model.add(Convolution2D(cl, 3, 3, activation='relu'))
    model.add(Flatten())
    for dl in dense_layers:
        model.add(Dense(dl))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer="adam")
    return model


def parse_input_csv_file(input_dir):
    """
    Load the input csv file and gather center, left, and right image pathes in long-format.
    Also, add the correction angle to left and right steering angles.
    """
    is_header = True
    input_file = 'driving_log.csv'
    steering_correction = 0.4

    X, y = [], []
    input_file = os.path.join(input_dir, input_file)
    local_dir = os.path.join(input_dir, 'IMG')  # path to IMG files
    with open(input_file) as csvfile:
        reader = csv.reader(csvfile)
        if is_header: next(reader, None) # if header skip first line
        for center, left, right, steering, throttle, brake, speed in reader:
            if float(speed) > 25: # only use stable drive samples
                X += [re.sub(r'.*IMG', local_dir, center.strip()),  # to support various folder path
                      re.sub(r'.*IMG', local_dir, left.strip()),    # replace path/to/IMG to local_dir
                      re.sub(r'.*IMG', local_dir, right.strip())]
                y += [float(steering),
                       float(steering) + steering_correction,
                       float(steering) - steering_correction]
    return np.array(X), np.array(y)


def split_data(X, y, train_thres=0.8, valid_thres=0.9, n_sample=None):
    """
    Split input data to training, validation and test set.
    """
    if not n_sample:
        n_sample = len(X)
    X_train, y_train = X[:int(n_sample*train_thres)], y[:int(n_sample*train_thres)]
    X_valid, y_valid = X[int(n_sample*train_thres):int(n_sample*valid_thres)], \
                       y[int(n_sample*train_thres):int(n_sample*valid_thres)]
    X_test, y_test = X[int(n_sample*valid_thres):], y[int(n_sample*valid_thres):]
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def flow_index(X, batch_size, shuffle):
    """
    Select indices for each batch.
    """
    batch_index = 0
    n = len(X)
    while 1:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0
        yield index_array[current_index: current_index + current_batch_size]


def random_h_shift_img(img, angle):
    """
    Shift (translate) input image randomly with adjusted angle.
    """
    h_shift_range = 20.0
    angle_range = 0.1  # angle per pixel = 0.005
    rows, cols, _ = img.shape

    dx = random.randint(-h_shift_range, h_shift_range)

    trans_mat = np.float32([[1, 0, dx], [0, 1, 0]])
    img = cv2.warpAffine(img, trans_mat, (cols, rows))

    angle = angle + angle_range * dx / h_shift_range

    return img, angle


def random_flip_img(img, angle):
    """
    Randomly flip image with probability of 0.5.
    """
    if random.uniform(0, 1) < 0.5:
        img = flip_axis(img, 1)
        angle = -angle
    return img, angle


def process_img(img_path, angle, augment):
    """
    Load images, and then generate additional data when augment=True.
    """
    target_size = None #(160, 320)

    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img)
    if augment:
        img, angle = random_h_shift_img(img, angle)
        img, angle = random_flip_img(img, angle)
    return img, angle


def image_data_generator(X, y, batch_size, shuffle=True, augment=True):
    """
    Generator to train model. When augment=True, the augmented images with adjusted angles
    are returned.
    """
    index_generator = flow_index(X, batch_size, shuffle)

    while 1:
        index_array = next(index_generator)

        batch_X, batch_y = [], []
        for i in index_array:
            image, steering = process_img(X[i], y[i], augment=augment)
            batch_X.append(image)
            batch_y.append(steering)

        yield np.array(batch_X), np.array(batch_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning Model')
    parser.add_argument('--epochs', metavar='N', type=int, nargs='?', default=1,
                        help='The number of epochs.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=256,
                        help='The batch size')
    parser.add_argument('--data_folder', type=str, nargs='?',
                        default='simulator/sample_data',
                        help='Path to dataset folder.')
    args = parser.parse_args()

    print("epochs = {}, batch_size = {}".format(args.epochs, args.batch_size))
    print("data_folder:", args.data_folder)

    X, y = parse_input_csv_file(args.data_folder)
    print("The number of total images:", X.shape[0])
    # split data to training, validation and test set (80%, 20%, 0%, respectively)
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_data(X, y, 0.8, 1.0)
    print("The number of training images:", len(X_train))
    print("The number of validation images:", len(X_valid))
    print("The number of test images:", len(X_test)) # we don't use test image

    input_image_shape = (160, 320, 3)
    model = get_model(shape=input_image_shape)

    # generator for training set
    g_train = image_data_generator(X_train, y_train, args.batch_size,
                             shuffle=True, augment=True)
    # generator for validation set
    g_valid = image_data_generator(X_valid, y_valid, args.batch_size,
                             shuffle=False, augment=False)
    model.fit_generator(g_train, samples_per_epoch=len(X_train), nb_epoch=args.epochs,
                        validation_data=g_valid, nb_val_samples=len(X_valid))
    model.save('model.h5')

    # To avoid occasional session exception
    K.clear_session()