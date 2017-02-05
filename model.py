import argparse
from datetime import datetime
import os

import numpy as np

#from keras.models import load_model

model = None
prev_image_array = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning Model')
    parser.add_argument('--epochs', metavar='N', type=int, nargs='?', default=10,
                        help='The number of epochs.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=256,
                        help='The batch size')
    parser.add_argument('--data_folder', type=str, nargs='?',
                        default='simulator/sample_data',
                        help='Path to dataset folder.')
    args = parser.parse_args()

    print("epochs = {}, batch_size = {}".format(args.epochs, args.batch_size))
    print("data_folder:", args.data_folder)
    print(os.listdir(args.data_folder))

    #model = load_model(args.model)

    # if args.image_folder != '':
    #     print("Creating image folder at {}".format(args.image_folder))
    #     if not os.path.exists(args.image_folder):
    #         os.makedirs(args.image_folder)
    #     else:
    #         shutil.rmtree(args.image_folder)
    #         os.makedirs(args.image_folder)
    #     print("RECORDING THIS RUN ...")
    # else:
    #     print("NOT RECORDING THIS RUN ...")
