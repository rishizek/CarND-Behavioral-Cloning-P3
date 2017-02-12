import argparse
import os, csv, re
import numpy as np
from datetime import datetime
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from keras import backend as K



model = None


def get_model(shape, load=False, checkpoint=None):
    """return the pre-trained model from file."""
    if load and checkpoint: return load_model(checkpoint)

    conv_layers, dense_layers = [32, 32, 64, 128], [1024, 512]

    model = Sequential()
    model.add(Flatten(input_shape=shape))
    # model.add(Convolution2D(32, 3, 3, activation='elu', input_shape=shape))
    # model.add(MaxPooling2D())
    # for cl in conv_layers:
    #     model.add(Convolution2D(cl, 3, 3, activation='elu'))
    #     model.add(MaxPooling2D())
    # model.add(Flatten())
    # for dl in dense_layers:
    #     model.add(Dense(dl, activation='elu'))
    #     model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer="adam")
    return model

def parse_input_data(input_dir):
    is_header = True
    input_file = 'driving_log.csv'
    steering_offset = 0.4

    X, y = [], []
    input_file = os.path.join(input_dir, input_file)
    local_dir = os.path.join(input_dir, 'IMG')  # path to IMG files
    with open(input_file) as csvfile:
        reader = csv.reader(csvfile)
        if is_header: next(reader, None) # if header skip first line
        for center, left, right, steering, throttle, brake, speed in reader:
            if float(speed) > 25: # only use stable drive samples
                X += [re.sub(r'.*IMG', local_dir, center.strip()),  # to support various folder path
                      re.sub(r'.*IMG', local_dir, left.strip()),    # remove path before IMG
                      re.sub(r'.*IMG', local_dir, right.strip())]
                y += [float(steering),
                       float(steering) + steering_offset,
                       float(steering) - steering_offset]
    return np.array(X), np.array(y)

def split_data(X, y, train_thres=0.8, valid_thres=0.9, n_sample=None):
    """split input data to training, validation and test set"""

    if not n_sample:
        n_sample = len(X)
    X_train, y_train = X[:int(n_sample*train_thres)], y[:int(n_sample*train_thres)]
    X_valid, y_valid = X[int(n_sample*train_thres):int(n_sample*valid_thres)], \
                       y[int(n_sample*train_thres):int(n_sample*valid_thres)]
    X_test, y_test = X[int(n_sample*valid_thres):], y[int(n_sample*valid_thres):]
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def flow_index(X, batch_size, shuffle):
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


def h_translate_img(img, angle, delta):
    return img, angle


def flip_img(img, angle):
    return img, -angle

def process_img(img_path, angle, augment):
    delta_h = 0.0
    target_size = None #(320, 160)
    if augment and 'center' in img_path:
        #print(img_path)
        pass

    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img)
    if augment:
        img, angle = h_translate_img(img, angle, delta_h)
        img, angle = flip_img(img, angle)
    return img, angle


def image_data_generator(X, y, batch_size, shuffle=True, augment=True):
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

    X, y = parse_input_data(args.data_folder)
    print(X.shape, y.shape)
    # split data to training, validation and test set (80%, 10%, 10%, respectively)
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_data(X, y, 0.8, 0.9)
    print(X_train.shape, X_valid.shape, X_test.shape)
    #model = load_model(args.model)
    image_shape = (160, 320, 3)
    model = get_model(shape=image_shape)


    g_train = image_data_generator(X_train, y_train, args.batch_size,
                             shuffle=True, augment=True)
    g_valid = image_data_generator(X_valid, y_valid, args.batch_size,
                             shuffle=False, augment=False)
    model.fit_generator(g_train, samples_per_epoch=len(X_train), nb_epoch=args.epochs,
                        validation_data=g_valid, nb_val_samples=len(X_valid))
    model.save('model.h5')

    # To avoid occasional session exception
    K.clear_session()