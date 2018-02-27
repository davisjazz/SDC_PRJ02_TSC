from utils_step0 import parse_args, parameters, dir_check, dir_create, data_load, channel, color_map
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

# # Parameter
# default_dir   = 'C:/Users/mo/home/_eSDC2_/_PRJ02_/_2_WIP/_1_forge/_testing_/'
# default_csv   = default_dir+'data/signnames.csv'
#
# # signnames
# signnames  = pd.read_csv(default_csv) # , index_col=0)
# class_id   = signnames.ClassId.tolist()
# sign_names = signnames.SignName.tolist()
# print(signnames)
# print(signnames['ClassId'])
# print(signnames['SignName'])


def main():
    # parameters and placeholders
    args = parse_args()
    flags = parameters()

    # load the dataset
    X_train, y_train, s_train, c_train = data_load(args.dtset, 'train.p')


    # # what's the shape of an traffic sign image?
    # label_shape = type(y_train.tolist()) # .shape  # [0].shape
    # print("label data shape  =", label_shape)
    # print()
    # print("label[0] =", y_train[:10])

    df = pd.DataFrame(y_train, columns=['labels']) #, index=range(43))
    values = df['labels'].value_counts().keys().tolist()
    counts = df['labels'].value_counts().tolist()

    toto = df['labels'].value_counts()
    toto = toto.sort_index(axis=0)
    #print(toto)

    plt.figure()
    toto.plot.hist(xlim=(-1, 43), color='#F0433A', stacked=True, bins=50)

    # Test OK - voir jupyter notebook

if __name__ == '__main__':
    main()