from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
from utils_step2 import dataVisu, showTrace
from utils_step3 import hFct, dataPPro, proAll, creaShow
from utils_step4 import jitShift, jitRot, jitCrop, barPrg, jitData, jitItall, jitListChart
from utils_step5 import tBoard
from utils_step6 import tSign
from utils_step7 import modArc
from utils_step8 import trainMod

import numpy as np
import random
import matplotlib.patches as patches, mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
import PIL
from PIL import Image


# Helper function: Show xSize*ySize images
# Visualize both the validation and the test accuracies
def showAccu1(ltLog,ltAccTst,ltAccVal,ltCost, cOlor = 'r', sTr = 'model1 - variation on learning rate', ratio = 1000):
    N = len(ltLog)
    # Plot both the validation and the test accuracies with various learning rates
    ltCost = [x * ratio for x in ltCost] 
    plt.scatter(ltAccTst,ltAccVal,s=ltCost, facecolors='none', edgecolors=cOlor)
    for i in range(N):
        if ltAccTst[i] < sorted(set(ltAccTst))[-2] and ltAccTst[i] > min(ltAccTst):
            plt.annotate(ltLog[i], (ltAccTst[i],ltAccVal[i]),  xycoords='data',
                    xytext=(-30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        elif ltAccTst[i] == min(ltAccTst):
            plt.annotate(ltLog[i], (ltAccTst[i],ltAccVal[i]),  xycoords='data',
                    xytext=(20, 0), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
        else:
            plt.annotate(ltLog[i], (ltAccTst[i],ltAccVal[i]),  xycoords='data',
                    xytext=(-60, 0), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
            
    plt.xlabel('test accuracy')
    plt.ylabel('validation accuracy')
    
    # Title
    plt.title(sTr,y=1.05)
    plt.show()


def showAccu2(ltScatter, sTr = 'model1', ratio = 2000):
    ltColor =  ['r','b','g','y','m','k'] #  ['r','b','g','y','m','k']
    recs, classes = [], []
    
    for lt,cOlor in zip(ltScatter,ltColor): # ltColor[:len(ltScatter)]):
        N = len(lt[0])
        # Plot both the validation and the test accuracies with various learning rates
        ltCost    = [x * ratio for x in lt[3]] # maxAccTst = sorted(set(lt[1]))[-1]
        for i in range(N):
            #plt.scatter(lt[1],lt[2],s=ltCost, facecolors='none', edgecolors=cOlor, linewidth='2')   #, marker='x'  
            plt.scatter(lt[1],lt[2],s=ltCost, facecolors=cOlor, edgecolors='y') 
            plt.annotate(lt[0][i], (lt[1][i],lt[2][i])) #, xytext=(-10, 0))
        recs.append(mpatches.Rectangle((0,0),1,1,fc=cOlor))
        classes.append(lt[-1])
        
    plt.legend(recs,classes, loc=4,fontsize=8)   
    plt.xlabel('test accuracy')
    plt.ylabel('validation accuracy')
    plt.grid() # (b=None, which=u'major', axis=u'both'
    
    fig = plt.gcf()
    fig.set_size_inches(15, 7)
    
    # Title
    plt.title(sTr,y=1.05)
    plt.show()


def main():
    args = parse_args()

    # Visualize both the validation and the test accuracies with various hyperparameters
    ltScatter, ltScat0, ltScat1, ltScat2, ltScat3, ltScat4 = [], [], [], [], [], [] 
    selection = 1

    if selection == 1:
        # model1 - List the validation and the test accuracies with various learning rates
        ltLog0    = ['1E-03','2E-03','9E-04','1E-04','8E-04','9.5E-04','8.5E-04']
        ltAccTst0 = [90.3,89.5,92.8,87.0,92.6,91.7,92.9]
        ltAccVal0 = [91.7,89.9,93.9,87.4,93.4,92.9,94.4]
        ltCost0   = [0.456,0.392,0.293,0.487,0.523,0.371,0.326]
        ltScat0   = [ltLog0,ltAccTst0,ltAccVal0,ltCost0,'learning rate']
        ltScatter = [ltScat0] #, ltScat1, ltScat2, ltScat3, ltScat4]
        showAccu(ltScatter, 'model1 - List the validation and the test accuracies with various learning rates')
    elif selection == 2:
        # model1 - List the validation and the test accuracies with various learning rates
        ltLog0    = ['1E-03','2E-03','9E-04','1E-04','8E-04','9.5E-04','8.5E-04']
        ltAccTst0 = [90.3,89.5,92.8,87.0,92.6,91.7,92.9]
        ltAccVal0 = [91.7,89.9,93.9,87.4,93.4,92.9,94.4]
        ltCost0   = [0.456,0.392,0.293,0.487,0.523,0.371,0.326]
        ltScat0   = [ltLog0,ltAccTst0,ltAccVal0,ltCost0,'learning rate']

        # model1 - List the validation and the test accuracies with various dropout rates
        ltLog1    = ['0.5','0.75','0.85','0.6','0.8','0.67','0.7']
        ltAccTst1 = [92.9,93.4,92.5,93.2,92.1,93.5,93.3]
        ltAccVal1 = [94.4,94.3,95.0,94.9,93.3,94.7,94.1]
        ltCost1   = [0.326,0.350,0.978,0.457,0.450,0.274,0.701]
        ltScat1   = [ltLog1,ltAccTst1,ltAccVal1,ltCost1,'dropout rate']

        # model1 - List the validation and the test accuracies with various proprocessed data
        ltLog2    = ['0RGB','0GRAY','0SHP','0HST','0CLAHE']
        ltAccTst2 = [94.7,94.5,92.0,91.5,91.8]
        ltAccVal2 = [94.9,96.3,93.1,93.2,92.0]
        cost2     = [0.492,0.279,0.493,0.473,0.511]
        ltScat2   = [ltLog2,ltAccTst2,ltAccVal2,cost2,'preprocessed']

        # model1 - List the validation and the test accuracies with various amount of jittered data (grayscale & centered normalized)
        ltLog3    = ['500','1000','1500','2000','2500','3000']
        ltAccTst3 = [ 91.8 , 94.6 , 95.2 , 95.1 , 95.7 , 94.2 ]
        ltAccVal3 = [ 92.0 , 95.8 , 97.1 , 97.0 , 97.1 , 97.6 ]
        cost3     = [ 0.511 , 0.312 , 0.164 , 0.163 , 0.161 , 0.103 ]
        ltScat3   = [ltLog3,ltAccTst3,ltAccVal3,cost3,'jittered: GRAY']   

        # model1 - List the validation and the test accuracies with various amount of jittered data (rgb & centered normalized)
        ltLog4    = ['500','1000','1500','2000','2500','3000']
        ltAccTst4 = [95.6,94.3,94.4,94.8,95.4,95.8]
        ltAccVal4 = [96.2,96.3,95.5,96.6,96.1,97.7]
        cost4     = [0.333,0.209,0.344,0.243,0.339,0.141]
        ltScat4   = [ltLog4,ltAccTst4,ltAccVal4,cost4,'jittered: RGB']

        # model1 - List the validation and the test accuracies with a leaky RELU and various amount of jittered data (centered normalized)
        ltLog5    = ['3e3xRGB','3e3xGRAY']
        ltAccTst5 = [96.1,95.3]
        ltAccVal5 = [95.7,97.4]
        cost5     = [0.285,0.143]
        ltScat5   = [ltLog5,ltAccTst5,ltAccVal5,cost5,'leakyRELU']

        ltScatter = [ltScat0, ltScat1, ltScat2, ltScat3, ltScat4, ltScat5]
        showAccu(ltScatter)

    elif selection == 3:
        # model1 - List the validation and the test accuracies
        ltLog0    = ['model1']
        ltAccTst0 = [95.8]
        ltAccVal0 = [97.7]
        ltCost0   = [0.141]
        ltScat0   = [ltLog0,ltAccTst0,ltAccVal0,ltCost0,'model 1']

        # model2 - List the validation and the test accuracies
        ltLog1    = ['model2']
        ltAccTst1 = [93] 
        ltAccVal1 = [94.5]
        ltCost1   = [0.777]
        ltScat1   = [ltLog1,ltAccTst1,ltAccVal1,ltCost1,'model 2']

        # model2b - List the validation and the test accuracies
        ltLog2    = ['model2b']
        ltAccTst2 = [94.8]
        ltAccVal2 = [96.2]
        cost2     = [0.386]
        ltScat2   = [ltLog2,ltAccTst2,ltAccVal2,cost2,'model 2b']

        # model2c - List the validation and the test accuracies
        ltLog3    = ['model2c']
        ltAccTst3 = [ 95.3 ]
        ltAccVal3 = [ 95.5 ]
        cost3     = [ 0.784 ]
        ltScat3   = [ltLog3,ltAccTst3,ltAccVal3,cost3,'model 2c']   

        # model2d - List the validation and the test accuracies
        ltLog4    = ['model2d']
        ltAccTst4 = [95.7]
        ltAccVal4 = [97.3]
        cost4     = [1.011]
        ltScat4   = [ltLog4,ltAccTst4,ltAccVal4,cost4,'model 2d']

        ltScatter = [ltScat0, ltScat1, ltScat2, ltScat3, ltScat4]

        showAccu(ltScatter, 'Comparison of various model results')


if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''