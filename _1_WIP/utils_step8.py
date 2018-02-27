from utils_step0 import parse_args, parameters, dir_check, dir_create, data_load, channel, color_map
from utils_step5 import layer_conv, layer_fcon, layer_flatten, evaluate, model_train, model_test, generator, early_stop
from utils_step6 import Model
from sklearn.utils import shuffle
import glob
import cv2
import numpy as np
from numpy import newaxis
import tensorflow as tf
import matplotlib.pyplot as plt


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
# def prediction(args, flags, logits, images_new):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver_new = tf.train.Saver()  # tf.train.import_meta_graph(flags.meta_graph)  # tf.train.Saver()
#         saver_new.restore(sess, tf.train.latest_checkpoint('./'))
#
#         if channel(images_new[0]) == 3:
#             x = flags.x3
#         else:
#             x = flags.x1
#
#         # predict the traffic sign
#         prediction = sess.run(tf.argmax(logits, 1), feed_dict={x: images_new, flags.keep_prob: 1.0})
#
#     return prediction.tolist()
def prediction(args, flags, logits, images_new):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_new = tf.train.Saver()  # tf.train.import_meta_graph(flags.meta_graph)
        saver_new.restore(sess, tf.train.latest_checkpoint('./'))

        if channel(images_new[0]) == 3:
            x = flags.x3
        else:
            x = flags.x1

        # predict the traffic sign
        prediction_top_1 = tf.argmax(logits, 1)
        prediction_top_5 = tf.nn.top_k(tf.nn.softmax(logits), args.top_k)
        prediction, prediction_top_k = sess.run([prediction_top_1, prediction_top_5], feed_dict={x: images_new, flags.keep_prob: 1.0})

    return prediction.tolist(), prediction_top_k


# Helper function: show the prediction
def predictions_show(args, images_new, labels_new, predictions):
    with open(args.file_csv) as f:
        f.readline()
        tuples = [line.strip().split(',') for line in f]
        sign_names = {int(t[0]): t[1] for t in tuples}

    for (image,label,prediction) in zip(images_new, labels_new, predictions):
        print('image.shape: {}'.format(image.shape))
        image = image.squeeze()
        annotation = "Actual:    %s\nPredicted: %s" % ( sign_names[label], sign_names[prediction] )
        fig = plt.figure(figsize=(1,1))
        plt.imshow(image)
        plt.annotate(annotation,xy=(0,0), xytext=(60,25), fontsize=12, family='monospace')
        plt.show()


# Helper function: calculate the accuracy of predictions the new images
def performance(labels_new, predictions):
    # predictions = predictions.tolist()
    success_rate = [ 1 if result else 0 for result in [ label == prediction for (label, prediction) in zip(labels_new, predictions) ]]
    try:
        return ( sum(success_rate) / len(success_rate) ) # * 100.0
    except ZeroDivisionError:
        return 0


def main():
    # parameters and placeholders
    args = parse_args()
    flags = parameters()

    # load new data set
    args.serie = '_imgOK_/' # '_serie01_/'
    images_new, labels_new = images_load(args)

    # preprocess data
    images_new = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images_new])
    images_new = (images_new - np.mean(images_new)) / np.std(images_new)
    images_new = images_new[..., newaxis]
    # print(' prepro images_new.shape : {}'.format(images_new.shape))
    # print()

    # predict on new images_new
    cnn    = Model()
    logits = cnn.model_1(flags, images_new[0])
    predictions, prediction_top_k = prediction(args, flags, logits, images_new)
    print(' predictions : {}'.format(predictions))
    print(' labels_new : {}'.format(labels_new))
    print()
    # print(' prediction_top_k : {}'.format((prediction_top_k)))
    # print()

    # measure the prediction success rate
    success_rate = performance(labels_new, predictions)
    print(' success_rate : {:.1%}'.format(success_rate))

    # # show predictions
    # predictions_show(args, images_new, labels_new, predictions)



if __name__ == '__main__':
    main()
