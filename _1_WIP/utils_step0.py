import argparse
import os
import pickle
import numpy as np
import tensorflow as tf

# Parameter
default_dir   = 'C:/Users/mo/home/_eSDC2_/_PRJ02_/_2_WIP/_1_forge/_testing_/'
default_data  = default_dir+'data/' # input('Enter the data path directory of the project data set')
default_image = default_dir+'data/images/'
default_csv   = default_dir+'data/signnames.csv'
default_dico  = default_dir+'data/dictionaries/data_dictionay.csv'
default_ppro  = default_dir+'data/preprocessed/'
default_jit   = default_dir+'data/jittered/'
default_addon = default_dir+'data/jittered/addon/'
default_full  = default_dir+'data/jittered/full/'
default_tb    = default_dir+'data/tboard/'
default_log   = default_dir+'logs/nn_logs/'
default_ckp   = default_dir+'logs/nn_logs/'
default_sprImg= default_dir+'data/tboard/_sp_valid_2144x2144.png'
default_sprTsv= default_dir+'data/tboard/_sp_valid_2144x2144.tsv'
default_newImg= 'C:/Users/mo/home/_eSDC2_/_PRJ02_/_2_WIP/_1_forge/_coding_/ownData/'

# Helper function: command-line / parse parameters
def parse_args():
    # y z
    parser = argparse.ArgumentParser(prog='traffic sign recognition', description='train a CNN to recognize traffic sign')
    parser.add_argument('-p', '--dir',    dest='dir', help='root directory path', action='store', type=str, default=default_dir)
    parser.add_argument('-d', '--dtset',  dest='dtset', help='data directory path', action='store', type=str, default=default_data)
    parser.add_argument('-i', '--img',    dest='image', help='image directory path', action='store', type=str, default=default_image)
    parser.add_argument('-c', '--csv',    dest='file_csv', help='csv file directory path', action='store', type=str, default=default_csv)
    parser.add_argument('-n', '--dico',   dest='dico', help='dictionary csv file directory path', action='store', type=str, default=default_dico)
    parser.add_argument('-g', '--png',    dest='png', help='file format', action='store', type=str, default='png')
    parser.add_argument('-o', '--cmap',   dest='cmap', help='colormaps', action='store', default=None)

    parser.add_argument('-r', '--ppro',   dest='ppro', help='preprocessed data directory path', action='store', type=str, default=default_ppro)
    parser.add_argument('-j', '--jit',    dest='jit', help='jittered data directory path', action='store', type=str, default=default_jit)
    parser.add_argument('-a', '--addon',  dest='addon', help='addon jittered data directory path', action='store', type=str, default=default_addon)
    parser.add_argument('-f', '--full',   dest='full', help='full jittered data directory path', action='store', type=str, default=default_full)

    parser.add_argument('-t', '--tboard', dest='tboard', help='tensorboard materials directory path', action='store', type=str, default=default_tb)
    parser.add_argument('-l', '--log',    dest='log', help='log directory path', action='store', type=str, default=default_log)
    parser.add_argument('-k', '--ckp',    dest='ckp', help='checkpoint directory path', action='store', type=str, default=default_ckp)

    parser.add_argument('-b', '--tab',    dest='tab', help='table size', action='store', type=list, default=[5,10])

    parser.add_argument('-q', '--class',  dest='n_classes', help='number of classes', action='store', type=int, default=43)
    parser.add_argument('-e', '--epoch',  dest='epochs', help='epochs', action='store', type=int, default=1) # 120)
    parser.add_argument('-m', '--rate',   dest='rate', help='learning rate', action='store', type=float, default=0.00085)
    parser.add_argument('-s', '--dropout',   dest='dropout', help='dropout rate', action='store', type=float, default=0.67)
    parser.add_argument('-u', '--bsize',  dest='batch_size', help='batch size', action='store', type=int, default=100)

    parser.add_argument('-v', '--sprImg', dest='sprImg', help='sprite image', action='store', type=str, default=default_sprImg)
    parser.add_argument('-w', '--sprTsv', dest='sprTsv', help='sprite label', action='store', type=str, default=default_sprTsv)
    parser.add_argument('-x', '--newImg', dest='new_image', help='new images directory path', action='store', type=str, default=default_newImg)
    parser.add_argument('-y', '--serie', dest='serie', help='new set images directory path', action='store', type=str, default='_serie01_/')

    args   = parser.parse_args()
    return args


# Helper function: parameters
class parameters(object):
    def __init__(self,
                 x1         = tf.placeholder(tf.float32, (None, 32, 32, 1)),
                 x3         = tf.placeholder(tf.float32, (None, 32, 32, 3)),
                 y          = tf.placeholder(tf.int32, (None)),
                 keep_prob  = tf.placeholder(tf.float32),
                 meta_graph = 'model.meta',
                 mu         = 0,
                 sigma      = 0.1 ):

        self.x1         = x1         # input = placeholder for grayscale image
        self.x3         = x3         # input = placeholder for RGB image
        self.y          = y          # label
        self.keep_prob  = keep_prob  # dropout rate
        self.meta_graph = meta_graph
        self.mu         = mu         # hyperparameters
        self.sigma      = sigma      # hyperparameters

    def x1(self):
        return self.x1

    def x3(self):
        return self.x3

    def y(self):
        return self.y

    def keep_prob(self):
        return self.keep_prob

    def meta_graph(self):
         return self.meta_graph

    def mu(self):
        return self.mu

    def sigma(self):
        return self.sigma


# Helper function: create directory tree
def dir_check(path):
    '''Create a folder if not present'''
    if not os.path.exists(path):
        os.makedirs(path)

def dir_create(path, dir_dictionary):
    dir_check(path)

    for dir_root in dir_dictionary['root']:                  # dir_root      = logs
        dir_check(path+dir_root+'/')                          # args.dir/logs
        try:
            for dir_subroot in dir_dictionary[ dir_root ]:   # dir_subroot   = nn_logs
                dir_check(path+dir_root+'/'+dir_subroot+'/')
        except:
            pass

# Helper function: load the dataset in memory
def data_load(args, file_pickled='train.p'):
    file = args.dtset + file_pickled
    try:
        with open(file, mode='rb') as f:
            data = pickle.load(f)
        return data['features'], data['labels'], data['sizes'], data['coords']
    except:
        raise IOError('the project data set are not found in the data directory')

# def chMap(image):
#     if image.shape[-1] == 3:
#         cMap ='rgb'
#         ch   = 3
#     elif image.shape[-1] == 32 or image.shape[-1] == 1:
#         cMap ='gray'
#         ch   = 1
#     else:
#         raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))
#     return cMap, ch

def channel(image):
    if image.shape[-1] == 3:
        ch   = 3
    elif image.shape[-1] == 32 or image.shape[-1] == 1:
        ch   = 1
    else:
        raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))
    return ch

def color_map(image):
    if image.shape[-1] == 3:
        cMap ='rgb'
    elif image.shape[-1] == 32 or image.shape[-1] == 1:
        cMap ='gray'
    else:
        raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))
    return cMap

def main():
    args = parse_args()

    # create directory tree
    dir_dict = {}
    dir_dict = {'root'  : ['data', 'logs'],\
                'data': ['images','tboard','preprocessed','jittered'],\
                'logs'  : ['nn_logs'] }
    dir_create(args.dir, dir_dict)

    # load the dataset
    X_train, y_train, s_train, c_train = data_load(args, 'train.p')
    X_valid, y_valid, s_valid, c_valid = data_load(args, 'valid.p')
    X_test, y_test, s_test , c_test    = data_load(args, 'test.p')

    # number of training, validating, testing examples
    n_train = len(X_train)
    n_valid = len(X_valid)
    n_test  = len(X_test)
    print("Number of training examples =", n_train)
    print("Number of validing examples =", n_valid)
    print("Number of testing examples  =", n_test)

    # what's the shape of an traffic sign image?
    image_shape = X_train[0].shape
    print("Image data shape  =", image_shape)

    # how many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(y_train))
    print("Number of classes =", n_classes)


if __name__ == '__main__':
    main()