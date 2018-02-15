from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
from utils_step2 import dataVisu, showTrace
from utils_step3 import hFct, dataPPro, proAll, creaShow
from utils_step4 import jitShift, jitRot, jitCrop, barPrg, jitData, jitItall, jitListChart
import pandas as pd
from sklearn.utils import shuffle
from shutil import copyfile
# import numpy as np
# from numpy import newaxis
# import pickle
# import csv, cv2
# import os
# import random
# import prettyplotlib as ppl
# import brewer2mpl
# import matplotlib.pyplot as plt

# Helper function: tensorboard
class tBoard(object):
    def __init__(self):
        pass
    
    def dataSprite(dataImg, dataLabel):
        '''Calculate the validation dataset lenght'''
        import math
        num0 = math.ceil(len(dataImg)**0.5)
        num0 *= num0
        # TB.E-V: outImg, outLabel
        outImg, outLabel = np.empty((num0,dataImg.shape[1],dataImg.shape[2],dataImg.shape[3])), np.empty((num0))
        outImg[:dataImg.shape[0]], outLabel[:dataLabel.shape[0]] = dataImg[:].copy(), dataLabel[:].copy()
        outImg[dataImg.shape[0]:], outLabel[dataLabel.shape[0]:] = dataImg[-1], dataLabel[-1]
        return outImg, outLabel

        
    def iNitb(X_valEV, embedding_size, embedding_input):
        # Combine all of the summary nodes into a single op
        merged = tf.summary.merge_all()
        # Setup a 2D tensor variable that holds embedding
        embedding  = tf.Variable(tf.zeros([len(X_valEV), embedding_size]), name="test_embedding") # 4489, embedding_size
        assignment = embedding.assign(embedding_input)        
        return merged, embedding, assignment


    def logWriter(sess):
        # Create a log writer. run 'tensorboard --logdir=./logs/nn_logs' ------
        #from datetime import datetime as dt
        now = dt.now()
        str0= now.strftime("%y%m%dx%H%M")
        str1= "./logs/nn_logs/" + str0 + "/"
        writer = tf.summary.FileWriter(str1, sess.graph) # for 0.8
        writer.add_graph(sess.graph)
        return str0, str1, writer

    # Embedding Visualization: configuration ---------------------------------- 
    def eVisu(sprImg,sprTsv,sprPath,LOGDIR,sIze,embedding,writer):
        '''TensorBoard: Embedding Visualization'''
        # Note: use the same LOG_DIR where you stored your checkpoint.
        inFileImg, inFileTvs = sprPath+sprImg, sprPath+sprTsv
        outFileImg, outFileTvs = LOGDIR+sprImg, LOGDIR+sprTsv      
        copyfile(inFileImg,outFileImg)
        copyfile(inFileTvs,outFileTvs)
        # 4. Format:
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        # 5. Add as much embedding as is necessary (Here we add only one)
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name         #embedding_var.name
        embedding_config.sprite.image_path = outFileImg
        # 6. Link this tensor to its labels (e.g. metadata file)
        embedding_config.metadata_path = outFileTvs
        # 7. Specify the width and height of a single thumbnail.
        embedding_config.sprite.single_image_dim.extend([sIze, sIze])
        # 8. Saves a configuration file that TensorBoard will read during startup
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
        return config, embedding_config

# Helper function: 


def main():
    args = parse_args()

    x = tBoard()


if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''