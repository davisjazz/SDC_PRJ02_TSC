from utils_step0 import parse_args, parameters, dir_check, dir_create, data_load, channel, color_map
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Helper function: convolutional layer
def layer_conv(flags, input, name='layer_1_', filter=5, size_in=3, size_out=6, padding='VALID', regularization=None, activation=None, leak=0.2):
    # create a convolutional layer: input = 32x32xsize_in, output = 28x28xsize_out
    with tf.name_scope(name+'conv'):
        shape_output = [filter, filter, size_in, size_out]
        w = tf.Variable(tf.truncated_normal(shape_output, mean=flags.mu, stddev=flags.sigma), name=name + 'w')
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name=name + 'b')
        x = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding=padding)

        # activation
        if activation is None:
            pass
        elif activation == 'relu':
            x = tf.nn.relu(tf.add(x, b))
        elif activation == 'leaky_relu':
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            x = f1 * tf.add(x, b) + f2 * abs(tf.add(x, b))

        # regularization
        if regularization is None:
            pass
        elif regularization == 'max_pool':
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

        return x


# Helper function: full connected layer
def layer_fcon(flags, input, name='layer_2_', size_in=3, size_out=6, regularization=None, activation=None, leak=0.2):
    # create a full connected layer: input = size_in, output = size_out
    with tf.name_scope(name+'fc'):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=flags.sigma), name=name + 'w')
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name=name + 'b')
        x = tf.add(tf.matmul(input, w), b)

        # activation
        if activation is None:
            pass
        elif activation == 'relu':
            x = tf.nn.relu(tf.add(x, b))
        elif activation == 'leaky_relu':
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            x = f1 * tf.add(x, b) + f2 * abs(tf.add(x, b))

        # regularization
        if regularization is None:
            pass
        elif regularization == 'dropout':
            x = tf.nn.dropout(x, flags.keep_prob)

        return x


# Helper function: flatten layer
def layer_flatten(input, name='layer_3_'):
    with tf.name_scope(name+'fl'):
        x = flatten(input)
    return x


# Helper function: evaluate the loss and accuracy of the model
def evaluate(args, flags, logits, images, labels, one_hot_y):
    prediction_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

    n_images       = len(images)
    accuracy_total = 0
    sess           = tf.get_default_session()

    if channel(images[0]) == 3:
        x = flags.x3
    else:
        x = flags.x1

    for offset in range(0, n_images, args.batch_size):
        batch_x, batch_y = images[offset:offset + args.batch_size], labels[offset:offset + args.batch_size]
        accuracy         = sess.run(accuracy_operation, feed_dict={x: batch_x, flags.y: batch_y, flags.keep_prob: args.dropout})
        accuracy_total  += (accuracy * len(batch_x))
    return accuracy_total / n_images


# Helper function: train the model
def model_train(args, flags, logits, images_train, labels_train, images_validation, labels_validation):
    one_hot_y = tf.one_hot(flags.y, 43)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.rate)
    training_operation = optimizer.minimize(loss_operation)

    if channel(images_train[0]) == 3:
        x = flags.x3
    else:
        x = flags.x1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_images = len(images_train)

        print('Training... rate: {}, epochs: {}, batch size: {}, dropout rate: {}'.format(args.rate, args.epochs, args.batch_size, args.dropout))
        print()
        for i in range(args.epochs):
            images_train, labels_train = shuffle(images_train, labels_train)
            for offset in range(0, n_images, args.batch_size):
                end = offset + args.batch_size
                batch_x, batch_y = images_train[offset:end], labels_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, flags.y: batch_y, flags.keep_prob: args.dropout})

            validation_accuracy = evaluate(args, flags, logits, images_validation, labels_validation, one_hot_y)
            print('epoch: {:3} | validation accuracy : {:.3f}'.format(i + 1, validation_accuracy))

        saver = tf.train.Saver()
        saver.save(sess, './model')
        print("Model saved")


# Helper function: test the model
def model_test(args, flags, logits, images_test, labels_test):
    one_hot_y = tf.one_hot(flags.y, 43)
    with tf.Session() as sess:
        saver_new = tf.train.Saver() # tf.train.import_meta_graph('./model.meta')
        saver_new.restore(sess, tf.train.latest_checkpoint('./'))
        accuracy_test = evaluate(args, flags, logits,  images_test, labels_test, one_hot_y)
        print("test accuracy = {:.3f}".format(accuracy_test))



def generator():
    pass


def early_stop():
    pass



def main():
    # parameters and placeholders
    args  = parse_args()
    flags = parameters()

    X_train, y_train, s_train, c_train = data_load(args.dtset, 'train.p')

    print('channel X_train[0] : {}'.format(channel(X_train[0])))

if __name__ == '__main__':
    main()
