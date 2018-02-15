from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
from utils_step2 import dataVisu, showTrace
from utils_step3 import hFct, dataPPro, proAll, creaShow
from utils_step4 import jitShift, jitRot, jitCrop, barPrg, jitData, jitItall, jitListChart
from utils_step5 import tBoard
from utils_step6 import tSign
from utils_step7 import modArc
from utils_step8 import trainMod
from utils_step9 import showAccu1, showAccu2

import pandas as pd
from sklearn.utils import shuffle
from shutil import copyfile

# Helper function: 
def report(X_train, y_train, X_valid, y_valid, X_test, y_test, X_valEV, y_valEV, X_testEV, y_testEV):
    # Initialization
    sess, x, y, ch, one_hot_y, keep_prob = tSign.iNit(cMap, n_classes)
    # Create model 
    logits, embedding_input, embedding_size = modArc.model1(x,ch,0,0.1,keep_prob)  #dropout)
    mod0 = 'model1'
    # train the model
    print()
    print('Training... {}: {}, {}: {}, {}: {}, {}: {}, {}: {}'.format('model', mod0,'rate',rate,'epochs',EPOCHS,'batch size',BATCH_SIZE,'keep_prob',dropout))
    print()
    # Create cost & accuracy function -
    lossOpe, trainingOpe = tSign.loss(logits, one_hot_y, rate, mod0)
    accuOpe = tSign.accuracy(logits, one_hot_y, mod0)
    # TensorBoard: initialization of the Embedding Visualization 
    merged, embedding, assignment = tBoard.iNitb(X_valEV, embedding_size, embedding_input)    
    # Initialize the training of the model
    metaModel, n_train = trainMod(sess).initTrain()    
    # Create a log writer. run 'tensorboard logdir=./logs/nn_logs'
    str0, str1, writer = tBoard.logWriter(sess)
    LOGDIR = str1    
    # Define the training function 
    ltTrain = [X_train,y_train]
    ltValid = [X_valEV,y_valEV]
    ltTb    = [sprImg,sprTsv,sprPath,LOGDIR,32,embedding]    
    # Train the model -
    trainMod(sess).modTrain(ltTrain,ltValid,ltTb,EPOCHS,BATCH_SIZE,trainingOpe,x,y,keep_prob,dropout,writer,assignment,merged,lossOpe,accuOpe,str1,metaModel,mod0)
    # Measure the accuracy:
    total_accuracy = trainMod.modMeasure(X_valid,y_valid,metaModel,str1,accuOpe,x,y,keep_prob)
    print("Validation Accuracy = {:.3f}".format(total_accuracy))
    
    total_accuracy = trainMod.modMeasure(X_test,y_test,metaModel,str1,accuOpe,x,y,keep_prob)
    print("Test Accuracy = {:.3f}".format(total_accuracy))



def main():
    args = parse_args()

    # Valuate some parameters
    n_classes  = 43
    rate       = 0.00085
    mu         = 0
    sigma      = 0.1
    EPOCHS     = 100
    BATCH_SIZE = 100 # 1 # 50 # 100
    dropout    = 0.67 # 0.67 # 0.25 # 0.5 # 0.75
    sprImg     = '_sp_valid_2144x2144.png' #'_sp_valid_2144x2144.png' 
    sprTsv     = '_sp_valid_2144x2144.tsv' # '_sp_valid_2144x2144.tsv'
    sprPath    = pathTb

    # Create data for the embedding visualization
    try:
        X_valEV, y_valEV   = tBoard.dataSprite(X_valid, y_valid, False)
        X_testEV, y_testEV = tBoard.dataSprite(X_test, y_test, False)
        X_valEV = np.float32(X_valEV)
        X_testEV = np.float32(X_testEV)
    except:
        pass

    # Train, Validate and Test the Model
    report(X_train, y_train, X_valid, y_valid, X_test, y_test, X_valEV, y_valEV, X_testEV, y_testEV)



if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''