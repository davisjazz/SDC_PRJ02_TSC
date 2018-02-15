# Predict the Sign Type for Each Image
from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1  import ocrLabel, indexClass, dataExplo
from utils_step2  import dataVisu, showTrace
from utils_step3  import hFct, dataPPro, proAll, creaShow
from utils_step4  import jitShift, jitRot, jitCrop, barPrg, jitData, jitItall, jitListChart
from utils_step5  import tBoard
from utils_step6  import tSign
from utils_step7  import modArc
from utils_step8  import trainMod
from utils_step9  import showAccu1, showAccu2
from utils_step10 import report

import glob
import matplotlib.image as mpimg
from PIL import Image

# Helper function: prediction function
def fctPrediction(myImg, pathCkp):
    ckptMeta = 'model1.ckpt.meta'
    image = myImg[0].squeeze()
    if image.shape[-1] == 3:  
        Ckp  = pathLog+pathCkp[0]+'/'  # with RGB images
        cMap ='rgb'
        ch   = 3
    elif image.shape[-1] == 32 or image.shape[-1] == 1:
        Ckp  = pathLog+pathCkp[1]+'/' # with GRAYSCALE images
        cMap ='gray'
        ch   = 1
    else:
        raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape = (None, 32, 32, ch))
        y = tf.placeholder(tf.uint8, shape = (None), name='label')
        keep_prob = tf.placeholder(tf.float32)
        logits, embedding_input, embedding_size = modArc.model1(x,ch,0,0.1,keep_prob)
        
    # Prediction
    with tf.Session(graph = graph) as sess:
        sess.run(tf.global_variables_initializer())
        metaModel = tf.train.import_meta_graph(Ckp+ckptMeta)
        metaModel.restore(sess, tf.train.latest_checkpoint(Ckp))
        myInference = tf.argmax(logits, 1) #tf.nn.softmax(
        myPrediction = sess.run(myInference, feed_dict={ x: myImg, keep_prob: 1 })

    showTrace2(myImg, myPrediction,1,5)
    return myPrediction, ch


# Helper function: calculate the accuracy for these 5 new images
def calAccu(myLabel,myPrediction):
    a1 = [1 if c else 0 for c in [ i1 == i2 for (i1,i2)   in zip(myLabel, myPrediction)] ]
    try:
        return (sum(a1)/len(a1))* 100 # print('score = {0:.0f}%'.format((sum(a1)/len(a1))* 100))
    except ZeroDivisionError:
        print("Can't divide by zero")


def main():
    args = parse_args()

    # test a Model on New Images
    ## own images - download, resize and store images into a list and convert it into an array
    myData  = pathData+'newData/_ownData_/_serie01_/'
    myImage, myLabel = [], []
    for i, myImg in enumerate(glob.glob(myData+'*.png')):
        myLabel.append(int(myImg[len(myData):len(myData)+2]))       # int(myImg[0:1]))  # -6:-4]))
        image = cv2.imread(myImg)
        image = cv2.resize(image,(32,32),interpolation = cv2.INTER_CUBIC)
        myImage.append(image)
    myImage = np.asarray(myImage)
    print('< myLabel > = {}'.format(myLabel))
    ## Own images - display original images
    showTrace2(myImage, myLabel,xSize=1, ySize=5)
    ## Own images - Standarize, normalize and display RGB data
    myImg1 = (myImage - np.mean(myImage))/np.std(myImage)
    showTrace2(myImg1,myLabel,xSize=1, ySize=5)
    ## Own images - Standarize, normalize and display grayscale data
    myImg2 = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in myImage ])
    myImg2 = (myImg2 - np.mean(myImg2))/np.std(myImg2)
    myImg2 = myImg2[..., newaxis]
    showTrace2(myImg2,myLabel,xSize=1, ySize=5)


    ## prediction with centered, normalized and jittered images
    pathCkp = ['171023x1030_F3000RGB_R85e-5_D67','170704x0014_F3000GRAY_R85e-5_D67']
    myPredictionRGB, ch = fctPrediction(myImg1, pathCkp)
    print(('<myPrediction> : {}').format(myPredictionRGB))
    #myPredictionGRAY, ch = fctPrediction(myImg2, pathCkp)
    #print(('<myPrediction> : {}').format(myPredictionGRAY))


    ## calculation of the accuracy for my own new images
    if ch == 3: 
        myPrediction = myPredictionRGB[:]
    else: 
        myPrediction = myPredictionGRAY[:]
    print('')
    myAccu = calAccu(myLabel,myPrediction)
    print('myAccu = {0:.0f}%'.format(myAccu))


if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''