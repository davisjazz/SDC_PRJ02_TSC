from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
from utils_step2 import dataVisu, showTrace
from utils_step3 import hFct, dataPPro, proAll, creaShow
from utils_step4 import jitShift, jitRot, jitCrop, barPrg, jitData, jitItall, jitListChart
from utils_step5 import tBoard
from utils_step6 import tSign

import pandas as pd
from sklearn.utils import shuffle
from shutil import copyfile

# Helper function: define the architecture
class modArc(tSign):
    def __init__(self):
        pass

    # Define architecture model1 
    def model1(x, ch, mu, sigma, keep_prob):  # Lenet5
       # Layer 1: Conv{In:32x32xch;Out:28x28x6} > Activ. > mxPooling{In:28x28x6;Out:14x14x6}
        x = tSign.conv_layer(x, 5, ch, 6, 'layer1', mu, sigma, 'VALID')
        # Layer 2: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        x = tSign.conv_layer(x, 5, 6, 16, 'layer2', mu, sigma, 'VALID')
        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten'):
            x = flatten(x)  # tf.reshape(x, [-1, n_input])
        # Layer 3: Fully Connected{In:400;Out:120} > Activ. > Dropout
        x = tSign.fc_layer(x, 400, 120, 'layer3', True, True, keep_prob)
        # Layer 4: Fully Connected{In:120;Out:84} > Activ. > Dropout
        x = tSign.fc_layer(x, 120, 84, 'layer4', True, True, keep_prob)
        # Layer 5: Fully Connected{In:84;Out:43}
        logits = tSign.fc_layer(x, 84, 43, 'layer5', False, False, keep_prob)
        # EMBBEDED VISUALIZER
        embedding_input = logits
        embedding_size  = 43
        return logits, embedding_input, embedding_size    
    
    
    # Define architecture model2 
    def model2(x, ch, mu, sigma, keep_prob):
       # Layer 1: Conv{In:32x32xch;Out:28x28x6} > Activ. > mxPooling{In:28x28x6;Out:14x14x6}
        # def conv_layer      (x,filter_size,size_in,size_out,nAme, mu, sigma,pAdding='VALID',maxPool=True)
        xL1 = tSign.conv_layer(x, 5, ch, 6, 'layer1', mu, sigma, 'VALID', True)
        # Layer 2: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL2 = tSign.conv_layer(xL1, 5, 6, 16, 'layer2', mu, sigma, 'VALID', True)        
        # Layer 3: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL3 = tSign.conv_layer(xL2, 5, 16, 400, 'layer3', mu, sigma, 'VALID', False)
        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten'):
            xF2 = flatten(xL2)
            tf.summary.histogram('xF2-xD2', xF2)  # Add histogram summaries
            xF3 = flatten(xL3)
            tf.summary.histogram('xF3-xD3', xF3)  # Add histogram summaries
            xFI = tf.concat([xF3,xF2], 1)
            tf.summary.histogram('xFI', xFI)  # Add histogram summaries
        # Dropout: 
        xDI = tf.nn.dropout(xFI, keep_prob)
        # Layer 4: Fully Connected{In:120;Out:84} > Activ. > Dropout
        xF4 = tSign.fc_layer(xDI, 800, 120, 'layer4', True, False, keep_prob)   # xFI.get_shape()[-1]
        # Layer 5: Fully Connected{In:120;Out:84} > Activ. > Dropout
        xF5 = tSign.fc_layer(xF4, 120, 84, 'layer5', True, False, keep_prob)
        # Layer 6: Fully Connected{In:84;Out:43}
        logits = tSign.fc_layer(xF5, 84, 43, 'layer6', False, False, keep_prob)
        # EMBBEDED VISUALIZER
        embedding_input = logits
        embedding_size  = 43
        return logits, embedding_input, embedding_size


    # Define architecture model2b 
    def model2b(x, ch, mu, sigma, keep_prob):
        # Layer 1: Conv{In:32x32xch;Out:28x28x6} > Activ. > mxPooling{In:28x28x6;Out:14x14x6}
        # def conv_layer      (x,filter_size,size_in,size_out,nAme, mu, sigma,pAdding='VALID',maxPool=True)
        xL1 = tSign.conv_layer(x, 5, ch, 6, 'layer1', mu, sigma, 'VALID', True)
        # Layer 2: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL2 = tSign.conv_layer(xL1, 5, 6, 16, 'layer2', mu, sigma, 'VALID', True)
        # Layer 3: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL3 = tSign.conv_layer(xL2, 5, 16, 400, 'layer3', mu, sigma, 'VALID', False)
        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten'):
            xF2 = flatten(xL2)
            tf.summary.histogram('xF2-xD2', xF2)  # Add histogram summaries
            xF3 = flatten(xL3)
            tf.summary.histogram('xF3-xD3', xF3)  # Add histogram summaries
            xFI = tf.concat([xF3,xF2], 1)
            tf.summary.histogram('xFI', xFI)  # Add histogram summaries
        # Layer 4: Fully Connected{In:120;Out:84} > Activ. > Dropout
        xF4 = tSign.fc_layer(xFI, 800, 120, 'layer4', True, True, keep_prob)   # xFI.get_shape()[-1]
        # Layer 5: Fully Connected{In:120;Out:84} > Activ. > Dropout
        xF5 = tSign.fc_layer(xF4, 120, 84, 'layer5', True, True, keep_prob)
        # Layer 6: Fully Connected{In:84;Out:43}
        logits = tSign.fc_layer(xF5, 84, 43, 'layer6', False, False, keep_prob)
        # EMBBEDED VISUALIZER
        embedding_input = logits
        embedding_size  = 43
        return logits, embedding_input, embedding_size
        
        
    # Define architecture model2c 
    def model2c(x, ch, mu, sigma, keep_prob):
        # Layer 1: Conv{In:32x32xch;Out:28x28x6} > Activ. > mxPooling{In:28x28x6;Out:14x14x6}
        # def conv_layer      (x,filter_size,size_in,size_out,nAme, mu, sigma,pAdding='VALID',maxPool=True)
        xL1 = tSign.conv_layer(x, 5, ch, 6, 'layer1', mu, sigma, 'VALID', True)
        # Layer 2: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL2 = tSign.conv_layer(xL1, 5, 6, 16, 'layer2', mu, sigma, 'VALID', True)
        # Layer 3: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL3 = tSign.conv_layer(xL2, 5, 16, 400, 'layer3', mu, sigma, 'VALID', False)
        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten'):
            xF2 = flatten(xL2)
            tf.summary.histogram('xF2-xD2', xF2)  # Add histogram summaries
            xF3 = flatten(xL3)
            tf.summary.histogram('xF3-xD3', xF3)  # Add histogram summaries
            xFI = tf.concat([xF3,xF2], 1)
            tf.summary.histogram('xFI', xFI)  # Add histogram summaries            
        # Layer 4: Fully Connected{In:120;Out:84} > Activ. > Dropout
        xF4 = tSign.fc_layer(xFI, 800, 120, 'layer4', True, True, keep_prob)   # xFI.get_shape()[-1]
        # Layer 5: Fully Connected{In:120;Out:43} > Activ. > Dropout
        logits = tSign.fc_layer(xF4, 120, 43, 'layer5', True, False, keep_prob)
        # EMBBEDED VISUALIZER
        embedding_input = logits
        embedding_size  = 43
        return logits, embedding_input, embedding_size

        
    # Define architecture model2d -------------------
    def model2d(x, ch, mu, sigma, keep_prob):
        # Layer 1: Conv{In:32x32xch;Out:28x28x6} > Activ. > mxPooling{In:28x28x6;Out:14x14x6}
        xL1 = tSign.conv_layer(x, 5, ch, 6, 'layer1', mu, sigma, 'VALID', True)
        # Layer 2: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL2 = tSign.conv_layer(xL1, 5, 6, 16, 'layer2', mu, sigma, 'VALID', True)
        # Layer 3: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL3 = tSign.conv_layer(xL2, 5, 16, 400, 'layer3', mu, sigma, 'VALID', False)
        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten'):
            xF2 = flatten(xL2)
            tf.summary.histogram('xF2-xD2', xF2)  # Add histogram summaries
            xF3 = flatten(xL3)
            tf.summary.histogram('xF3-xD3', xF3)  # Add histogram summaries
            xFI = tf.concat([xF3,xF2], 1)
            tf.summary.histogram('xFI', xFI)  # Add histogram summaries
        # Dropout: 
        xDI = tf.nn.dropout(xFI, keep_prob)
        # Layer 4: Fully Connected{In:120;Out:84} > Activ. > Dropout
        logits = tSign.fc_layer(xDI, 800, 43, 'layer4', False, False, keep_prob)   # xFI.get_shape()[-1]
        # EMBBEDED VISUALIZER
        embedding_input = logits
        embedding_size  = 43
        return logits, embedding_input, embedding_size

    
    # Define architecture model3 
    def model3(x, ch, mu, sigma, keep_prob):
        # Layer 1: Conv{In:32x32xch;Out:32x32xch}
        #def conv_layer(x, 1, ch, 3, 'layer1', mu, sigma, 'SAME',maxPool=True, layer0=False, tRace=tRace1, msg=msg2):
        xL1 = tSign.conv_layer(x, 1, ch, 3, 'layer1', mu, sigma, 'SAME', False,layer0=False)
        # Layer 2: Conv{In:32x32xch;Out:32x32x32}
        xL2 = tSign.conv_layer(xL1, 5, 3, 32, 'layer2', mu, sigma, 'SAME', False)
        # Layer 3: Conv{In:32x32x32;Out:32x32x32} > Activ. > mxPooling{In:32x32x32;Out:16x16x32}
        xL3 = tSign.conv_layer(xL2, 5, 32, 32, 'layer3', mu, sigma, 'SAME', True)
        # Dropout: 
        xD1 = tf.nn.dropout(xL3, keep_prob)
        # Layer 4: Conv{In:16x16x32;Out:16x16x64} > Activ. > mxPooling{In:16x16x64;Out:16x16x64}
        xL4 = tSign.conv_layer(xD1, 5, 32, 64, 'layer4', mu, sigma, 'SAME', False)
        # Layer 5: Conv{In:16x16x64;Out:16x16x64} > Activ. > mxPooling{In:16x16x64;Out:8x8x16}
        xL5 = tSign.conv_layer(xL4, 5, 64, 64, 'layer5', mu, sigma, 'SAME', True)
        # Dropout: 
        xD2 = tf.nn.dropout(xL5, keep_prob)
        # Layer 6: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL6 = tSign.conv_layer(xD2, 5, 64, 128, 'layer6', mu, sigma, 'SAME', False)
        # Layer 7: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL7 = tSign.conv_layer(xL6, 5, 128, 128, 'layer7', mu, sigma, 'SAME', True)
        # Dropout: 
        xD3 = tf.nn.dropout(xL7, keep_prob)

        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten'):
            xF1 = flatten(xD1)   # tf.reshape(xD1, [-1, n_input])
            xF2 = flatten(xD2)
            xF3 = flatten(xD3)
            xFI = tf.concat([xF1,xF2,xF3], 1)
        # Layer 9: Fully Connected{In:120;Out:84} > Activ. > Dropout
        xF9 = tSign.fc_layer(xFI, 14336, 1024, 'layer9', True, True, keep_prob)   # xFI.get_shape()[-1]
        # Layer 10: Fully Connected{In:120;Out:84} > Activ. > Dropout
        xF10 = tSign.fc_layer(xF9, 1024, 1024, 'layer10', True, True, keep_prob)
        # Layer 11: Fully Connected{In:84;Out:43}
        logits = tSign.fc_layer(xF10, 1024, 43, 'layer11', False, False, keep_prob)
        # EMBBEDED VISUALIZER
        embedding_input = logits
        embedding_size  = 43
        return logits, embedding_input, embedding_size
    
    
    # Define architecture model4 
    def model4(x, ch, mu, sigma, keep_prob, tRace=tRace1, msg=msg2):
        # INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC 

        # < 1 > CONV -> RELU -> CONV -> RELU -> POOL 
        # Layer 1: Conv{In:32x32xch;Out:28x28x6} > Activ. > {Out:28x28x6}
        # note: (w - f + 2*p)/s +1  -> 28 = (32 - 5 + 2*0)/1 + 1
        xL1 = tSign.conv_layer(x, 5, ch, 32, 'layer1', mu, sigma, 'VALID',maxPool=False)
        # Layer 2: Conv{In:28x28x6;Out:24x24x16} > Activ. > mxPooling{In:24x24x16;Out:12x12x16}
        xL2 = tSign.conv_layer(xL1, 3, 32, 32, 'layer2', mu, sigma, 'VALID',maxPool=True)
        # < 2 > CONV -> RELU -> CONV -> RELU -> POOL 
        # Layer 3: Conv{In:12x12x16;Out:8x8x32} > Activ. > {Out:8x8x32}
        xL3 = tSign.conv_layer(xL2, 3, 32, 64, 'layer3', mu, sigma, 'VALID',maxPool=False)
        # Layer 4: Conv{In:8x8x32;Out:4x4x64} > Activ. > mxPooling{In:4x4x64;Out:2x2x64}
        xL4 = tSign.conv_layer(xL3, 3, 64, 128, 'layer4', mu, sigma, 'VALID',maxPool=True)
        # < 3 > CONV -> RELU -> CONV -> RELU -> POOL 
        # Layer 5: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL5 = tSign.conv_layer(xL4, 2, 128, 512, 'layer3', mu, sigma, 'VALID',maxPool=False)
        # Layer 6: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        xL6 = tSign.conv_layer(xL5, 2, 512, 1024, 'layer4', mu, sigma, 'VALID',maxPool=True)

        # < 1 > FC -> RELU
        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten1'):
            xF2 = flatten(xL2)
            tf.summary.histogram('xF2-xD2', xF2)  # Add histogram summaries
            xF3 = flatten(xL3)
            tf.summary.histogram('xF3-xD3', xF3)  # Add histogram summaries
            xFI = tf.concat([xF3,xF2], 1)
            tf.summary.histogram('xFI', xFI)  # Add histogram summaries
            # Activation
            actxFI = tf.nn.relu(xFI) #act = tf.nn.relu(conv + b)
            tf.summary.histogram('act_xFI', actxFI) # Add histogram summaries

        # < 2 > FC -> RELU
        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten2'):
            xF4 = flatten(xL4)
            tf.summary.histogram('xF4-xD4', xF4)  # Add histogram summaries
            xF5 = flatten(xL5)
            tf.summary.histogram('xF5-xD5', xF5)  # Add histogram summaries
            xFII = tf.concat([xF5,xF4], 1)
            tf.summary.histogram('xFII', xFII)  # Add histogram summaries
            # Activation
            actxFII = tf.nn.relu(xFII) #act = tf.nn.relu(conv + b)
            tf.summary.histogram('act_xFII', actxFII) # Add histogram summaries
        # Dropout:
        xFIII = tf.concat([actxFII,actxFI], 1)
        xDI = tf.nn.dropout(xFIII, keep_prob)
        # Layer 8: Fully Connected{In:19808;Out:9904} > Activ. > Dropout
        xF8 = tSign.fc_layer(xDI, 19808, 9904, 'layer8', True, False, keep_prob)   # xFI.get_shape()[-1]
        # Layer 9: Fully Connected{In:9904;Out:1238} > Activ. > Dropout
        xF9 = tSign.fc_layer(xF8, 9904, 1238, 'layer9', True, False, keep_prob)   # xFI.get_shape()[-1]
        # Layer 11: Fully Connected{In:1238;Out:619} > Activ. > Dropout
        xF11 = tSign.fc_layer(xF9, 1238, 619, 'layer11', True, False, keep_prob)   # xFI.get_shape()[-1]
        # Layer 12: Fully Connected{In:619;Out:120} > Activ. > Dropout
        xF12 = tSign.fc_layer(xF11, 619, 120, 'layer12', True, False, keep_prob)   # xFI.get_shape()[-1]
        # Layer 13: Fully Connected{In:120;Out:84} > Activ. > Dropout
        xF13 = tSign.fc_layer(xF12, 120, 84, 'layer13', True, False, keep_prob)
        # Layer 14: Fully Connected{In:84;Out:43}
        logits = tSign.fc_layer(xF13, 84, 43, 'layer14', False, False, keep_prob)
        # EMBBEDED VISUALIZER
        embedding_input = logits
        embedding_size  = 43
        return logits, embedding_input, embedding_size


def main():
    args = parse_args()


if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''