from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
from utils_step2 import dataVisu, showTrace
from utils_step3 import hFct, dataPPro, proAll, creaShow
from utils_step4 import jitShift, jitRot, jitCrop, barPrg, jitData, jitItall, jitListChart
from utils_step5 import tBoard
import pandas as pd
from sklearn.utils import shuffle
from shutil import copyfile

# Helper function: classifier
class tSign(object):
    def __init__(self):
        pass
    
    # In[step2.5]: Preliminaries - initialization
    def iNit(cMap, n_classes):
        # Remove previous Tensors and Operations
        tf.reset_default_graph()
        sess = tf.InteractiveSession() # sess = tf.Session()
        # Setup placeholders: features and labels       
        if cMap =='rgb':
            ch = 3
        elif cMap =='gray':
            ch = 1
        else:
            raise ValueError('Current cMap:',cMap,'. cMap should be ''rgb'' or ''gray''')
            
        x = tf.placeholder(tf.float32, (None, 32, 32, ch), name='input')  
        y = tf.placeholder(tf.uint8, (None), name='label') # y = tf.placeholder(tf.int32, (None, len(y_train)))
        # One-Hot
        one_hot_y = tf.one_hot(y, n_classes)
        # Add dropout to input and hidden layers
        keep_prob = tf.placeholder(tf.float32) # probability to keep units
        # Add image summary
        tf.summary.image('input', x, 8)             
        return sess, x, y, ch, one_hot_y, keep_prob
          
            
    # In[step2.5]: helper functions - conv_layer, fc_layer
    # conv_layer: Build a convolutional layer ---------------------------------
    def conv_layer(input,filter_size,size_in,size_out,nAme="conv", mu=0, sigma=0.1,pAdding='VALID',maxPool=True, aCtivation='relu', leak=0.2):
        with tf.name_scope(nAme):
            # Layer: Convolutional. Input = 32x32xsize_in. Output = 28x28xsize_out.
            shape0 = [filter_size, filter_size, size_in, size_out]
            w = tf.Variable(tf.truncated_normal(shape0, mean = mu, stddev = sigma), name=nAme+"W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name=nAme+"B")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding=pAdding)

            # Activation
            if aCtivation =='relu':
                act  = tf.nn.relu(tf.add(conv, b)) #act = tf.nn.relu(conv + b)
                str9 = 'RELU'
            else:
                f1  = 0.5 * (1 + leak)
                f2  = 0.5 * (1 - leak)
                act = f1 * tf.add(conv, b) + f2 * abs(tf.add(conv, b))  
                str9 = 'LEAKY RELU'
            
            # Add histogram summaries for weights and biases
            tf.summary.histogram(nAme+"_weights", w)
            tf.summary.histogram(nAme+"_biases", b)
            tf.summary.histogram(nAme+"_activations", act)
            
            if maxPool: 
                # Pooling. Input = 28x28xsize_out. Output = 14x14xsize_out.
                output = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=pAdding)
            else:
                output = act          
        return output    


    # fc_layer: Build a full connected layer ----------------------------------
    def fc_layer(input, size_in, size_out, nAme="fc", act = True, drop= True, keep_prob = tf.placeholder(tf.float32), aCtivation='relu', leak=0.2):       
        with tf.name_scope(nAme):
            # Layer: Convolutional. Input = size_in. Output = size_out.
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name=nAme+"W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name=nAme+"B")
            x = tf.add(tf.matmul(input, w), b)
            # Add histogram summaries for weights and biases
            tf.summary.histogram(nAme+"_weights", w)
            tf.summary.histogram(nAme+"_biases", b)

            if act: # Activation and histogram summaries:
                if aCtivation =='relu':
                    x = tf.nn.relu(x)
                    str9 = 'RELU'
                else:
                    f1  = 0.5 * (1 + leak)
                    f2  = 0.5 * (1 - leak)
                    x   = f1 * x + f2 * abs(x)  
                    str9 = 'LEAKY RELU'
                tf.summary.histogram(nAme+"_activations", x)
            if drop: # Dropout
                x = tf.nn.dropout(x, keep_prob)
            return x

    # Define cost function ----------------------------------------------------
    def loss(logits, one_hot_y, rate, mod0):
        with tf.name_scope("cost"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y, name="xent")  
            #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, one_hot_y) #<-- error msg
            lossOpe = tf.reduce_mean(cross_entropy, name="loss")
            optimizer = tf.train.AdamOptimizer(learning_rate = rate, name="optAdam")
            trainingOpe = optimizer.minimize(lossOpe, name="optMin")
            # Add scalar summary for loss (cost) tensor
            tf.summary.scalar(mod0+'_loss', lossOpe)        
        return lossOpe, trainingOpe


    # Define accuracy fct -----------------------------------------------------
    def accuracy(logits, one_hot_y, mod0):
        with tf.name_scope("accuracy"):
            correctPrd = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
            accuOpe = tf.reduce_mean(tf.cast(correctPrd, tf.float32))
            # Add scalar summary for accuracy tensor
            tf.summary.scalar(mod0+'_accuracy', accuOpe)        
        return accuOpe


# Helper function: 


def main():
    args = parse_args()


if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''