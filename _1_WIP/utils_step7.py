from utils_step0 import parse_args, parameters, dir_check, dir_create, data_load, channel, color_map
from utils_step5 import layer_conv, layer_fcon, layer_flatten, evaluate, model_train, model_test
from utils_step6 import Model
from sklearn.utils import shuffle


def main():
    # parameters and placeholders
    args = parse_args()
    flags = parameters()

    # load and shuffle data
    X_test, y_test, s_test , c_test    = data_load(args, 'test.p')
    X_test, y_test = shuffle(X_test, y_test)

    # build and train the model
    cnn    = Model()
    logits = cnn.model_1(flags, X_test[0])
    model_test(args, flags, logits, X_test, y_test)


if __name__ == '__main__':
    main()
