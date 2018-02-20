from utils_step0 import parse_args, parameters, dir_check, dir_create, data_load, channel, color_map
# from utils_step1 import ocrLabel, indexClass, dataExplo
# from utils_step2 import dataVisu, showTrace
# from utils_step3 import hFct, dataPPro, proAll, creaShow
# from utils_step4 import jitShift, jitRot, jitCrop, barPrg, jitData, jitItall, jitListChart
from utils_step5 import layer_conv, layer_fcon, layer_flatten, evaluate, model_train
import pandas as pd
from sklearn.utils import shuffle
from shutil import copyfile
import numpy as np
import tensorflow as tf
from datetime import datetime as dt


# Helper function: create a CNN model
class Model(object):
    def __init__(self):
        pass

    def model_lenet(self, flags, image):
        '''
        Layer 1: Conv{In:32x32xchannel;Out:28x28x6} > Activ. > mxPooling{In:28x28x6;Out:14x14x6}
        Layer 2: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        Flatten: Input = 5x5xsize_out. Output = 400.
        Layer 3: Fully Connected{In:400;Out:120} > Activ. > Dropout
        Layer 4: Fully Connected{In:120;Out:84} > Activ. > Dropout
        Layer 5: Fully Connected{In:84;Out:43}
        '''
        if channel(image) == 3:
            model = layer_conv(flags, flags.x3, name='layer_1_', filter=5, size_in=3, size_out=6, padding='VALID', regularization='max_pool', activation='relu')
        else:
            model = layer_conv(flags, flags.x1, name='layer_1_', filter=5, size_in=1, size_out=6, padding='VALID', regularization='max_pool', activation='relu')
        model = layer_conv(flags, model, name='layer_2_', filter=5, size_in=6, size_out=16, padding='VALID', regularization='max_pool', activation='relu')
        model = layer_flatten(model, name='layer_3_')
        model = layer_fcon(flags, model, name='layer_4_', size_in=400, size_out=120, regularization='dropout', activation='relu')
        model = layer_fcon(flags, model, name='layer_5_', size_in=120, size_out=84, regularization='dropout', activation='relu')
        model = layer_fcon(flags, model, name='layer_6_', size_in=84, size_out=43, regularization=None, activation=None)

        return model


def main():
    # parameters and placeholders
    args = parse_args()
    flags = parameters()

    # load and shuffle data
    X_train, y_train, s_train, c_train = data_load(args, 'train.p')
    X_valid, y_valid, s_valid, c_valid = data_load(args, 'valid.p')
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)

    # build and train the model
    cnn    = Model()
    logits = cnn.model_lenet(flags, X_train[0])
    model_train(args, flags, logits, X_train, y_train, X_valid, y_valid)

    # evaluate the model
    # compile the model


if __name__ == '__main__':
    main()
