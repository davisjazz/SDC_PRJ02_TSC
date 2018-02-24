from utils_step0 import parse_args, parameters, dir_check, dir_create, data_load, channel, color_map
from utils_step5 import layer_conv, layer_fcon, layer_flatten, evaluate, model_train, model_test
from utils_step6 import Model
from sklearn.utils import shuffle
import glob
import cv2
import numpy as np
from numpy import newaxis
import tensorflow as tf


# Helper function: load my own images in memory
def images_load(args):
    path = args.new_image + args.serie
    images, labels = [], []
    for i, image in enumerate(glob.glob(path + '*.png')):
        labels.append(int(image[len(path):len(path) + 2]))
        image = cv2.imread(image)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        images.append(image)
    images = np.asarray(images)
    return images, labels


# Helper function: make predictions the new images
def prediction(args, flags, logits, images_new):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_new = tf.train.Saver()  # tf.train.import_meta_graph(flags.meta_graph)  # tf.train.Saver()
        saver_new.restore(sess, tf.train.latest_checkpoint('./'))

        if channel(images_new[0]) == 3:
            x = flags.x3
        else:
            x = flags.x1

        # predict the traffic sign
        prediction = sess.run(tf.argmax(logits, 1), feed_dict={x: images_new, flags.keep_prob: 1.0})

    return prediction


# Helper function: show the prediction
def prediction_show():
    '''
    # X_new_images_p = preprocess(X_new_images)
    # y_new_images = [33, 17, 27, 3, 2, 14, 11, 18, 13, 28, 38, 40]
    # prediction = sess.run(tf.argmax(logits, 1), feed_dict={x: X_new_images_p})

    for (i,v) in enumerate(prediction):
        annotation = "Actual:    %s\nPredicted: %s" % (
            sign_names[y_new_images[i]], sign_names[v])
        fig = plt.figure(figsize=(1,1))
        plt.imshow(X_new_images[i])
        plt.annotate(annotation,xy=(0,0), xytext=(60,25), fontsize=12, family='monospace')
        plt.show()
    '''
    pass


def generator():
    pass


def early_stop():
    pass

def main():
    # parameters and placeholders
    args = parse_args()
    flags = parameters()

    # load new data set
    args.serie = '_serie01_/'
    images_new, labels_new = images_load(args)

    # preprocess data
    images_new = (images_new - np.mean(images_new)) / np.std(images_new)
    # print(' prepro images_new.shape : {}'.format(images_new.shape))
    # print()

    # predict on new images_new
    cnn    = Model()
    logits = cnn.model_1(flags, images_new[0])
    prediction_test = prediction(args, flags, logits, images_new)
    print(' prediction : {}'.format(prediction_test))
    print(' labels_new : {}'.format(labels_new))


if __name__ == '__main__':
    main()
