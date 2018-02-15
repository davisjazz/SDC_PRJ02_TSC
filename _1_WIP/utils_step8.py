from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
from utils_step2 import dataVisu, showTrace
from utils_step3 import hFct, dataPPro, proAll, creaShow
from utils_step4 import jitShift, jitRot, jitCrop, barPrg, jitData, jitItall, jitListChart
from utils_step5 import tBoard
from utils_step6 import tSign
from utils_step7 import modArc

import pandas as pd
from sklearn.utils import shuffle
from shutil import copyfile

# Helper function: Train, Validate and Test the Model
class trainMod(object):
    def __init__(self, sess):
        self.sess = sess
    
    # Initialization fo the training of the model 
    def initTrain(self,tRace=tRace1, msg=msg2):
        # Initiate the Saves and restores mechanisum 
        metaModel = tf.train.Saver()
        # Initialize all variables 
        self.sess.run(tf.global_variables_initializer())
        n_train = len(X_train)
        return metaModel, n_train


    def modTrain(self,ltTrain,ltValid,ltTb,EPOCHS,BATCH_SIZE,trainingOpe,x,y,keep_prob,dropout,writer,assignment,merged,lossOpe,accuOpe,str1,metaModel,mod0,tRace=tRace1, msg=msg2):
        self.sess.run(tf.global_variables_initializer())
        n_train = len(ltTrain[0])
        n_valid = len(X_valid)
        total_cost, total_accuracy = 0, 0
        
        for i0 in range(EPOCHS):
            for start, end in zip(range(0, n_train, BATCH_SIZE), range(BATCH_SIZE, n_train+1, BATCH_SIZE)):
                xBatch, yBatch = ltTrain[0][start:end], ltTrain[1][start:end]
                self.sess.run(trainingOpe, feed_dict={x: xBatch, y: yBatch, keep_prob: dropout}) # dropout})
                #sess.run(assignment, feed_dict={x: xBatch, y: yBatch, keep_prob: 0.5})
        
            # TensorBoard: Embedding Visualization 
            config, embedding_config = tBoard.eVisu(ltTb[0],ltTb[1],ltTb[2],ltTb[3],ltTb[4],ltTb[5],writer)
                       
            for start, end in zip(range(0, n_valid, BATCH_SIZE), range(BATCH_SIZE, n_valid+1, BATCH_SIZE)):
                xBatch, yBatch = X_valid[start:end], y_valid[start:end]
                cost, accuracy = self.sess.run([lossOpe, accuOpe], feed_dict={x: xBatch, y: yBatch, keep_prob: 1})
                total_cost     += (cost     * len(xBatch))
                total_accuracy += (accuracy * len(xBatch))
            
            total_cost     = total_cost / n_valid
            total_accuracy = total_accuracy / n_valid
            
            summary = self.sess.run(merged, feed_dict={x: X_valid, y: y_valid, keep_prob: 1})
                
            #metaModel.save(sess, str1)
            if not os.path.exists(str1):
                os.makedirs(str1)
            metaModel.save(self.sess, os.path.join(str1, mod0+'.ckpt')) #, i0)
        
            # Write summary
            writer.add_summary(summary, i0)
        
            # Report the accuracy
            print('Epoch: {:3} | cost : {:.3f} | Val.accu : {:.3f}'.format(i0, total_cost, total_accuracy)) #| asgnVal : {} ... ,asgnVal))


    # Measure the test accuracy:
    def modMeasure(dataImg,dataLabel,metaModel,str1,accuOpe,x,y,keep_prob):
        with tf.Session() as sess:    
            ## Step 13 - Need to initialize all variables
            sess.run(tf.global_variables_initializer())
            ## Step 14 - Evaluate the performance of the model on the test set
            metaModel.restore(sess, tf.train.latest_checkpoint(str1)) #pathMod))
            n_data = len(dataImg)
            total_accuracy = 0

            for start, end in zip(range(0, n_data, BATCH_SIZE), range(BATCH_SIZE, n_data+1, BATCH_SIZE)):
                xBatch, yBatch = dataImg[start:end], dataLabel[start:end]
                accuracy = sess.run(accuOpe, feed_dict={x: xBatch, y: yBatch, keep_prob: 1})
                total_accuracy += (accuracy * len(xBatch))
            total_accuracy = total_accuracy / n_data
            return total_accuracy



def main():
    args = parse_args()


if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''