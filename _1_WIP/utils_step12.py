# Output Top 5 Softmax Probabilities For Each Image Found on the Web
from utils_step0  import parse_args, dir_check, dir_create, data_load, chMap
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
from utils_step11 import fctPrediction, calAccu

import glob
import matplotlib.image as mpimg
from PIL import Image

# Helper function: 
def topKPrediction(myImg, pathCkp, k = 5, tRace=tRace1, msg=msg1):
    ckptMeta = 'model1.ckpt.meta'
    if tRace:print(msg,' | [{:5}] {}: {}'.format('fctPrediction','<ckptMeta>',ckptMeta))
        
    image = myImg[0].squeeze()
    if image.shape[-1] == 3:  
        Ckp  = pathLog+pathCkp[0]+'/'  # with RGB images
        cMap ='rgb'
        ch   = 3      
        if tRace:print(msg,' | [{:5}] {}: {}'.format('step3.2.1','<RGB>',Ckp[len(pathLog):-1]))
    elif image.shape[-1] == 32 or image.shape[-1] == 1:
        Ckp  = pathLog+pathCkp[1]+'/' # with GRAYSCALE images
        cMap ='gray'
        ch   = 1
        if tRace:print(msg,' | [{:5}] {}: {}'.format('step3.2.1','<GRAY>',Ckp[len(pathLog):-1]))
    else:
        raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape = (None, 32, 32, ch))
        y = tf.placeholder(tf.uint8, shape = (None), name='label')
        keep_prob = tf.placeholder(tf.float32)
        logits, embedding_input, embedding_size = modArc.model1(x,ch,0,0.1,keep_prob)
        tK0 = tf.nn.top_k(tf.nn.softmax(logits), k)
        
    # Prediction
    with tf.Session(graph = graph) as sess:
        sess.run(tf.global_variables_initializer())
        metaModel = tf.train.import_meta_graph(Ckp+ckptMeta)
        metaModel.restore(sess, tf.train.latest_checkpoint(Ckp))
        tK1 = sess.run(tK0, feed_dict={ x: myImg, keep_prob: 1 })
        
        return tK1


# Helper function: Show xSize*ySize images
def showTrace3(dataImg, dataLabel, myImg, myLabel, top_K ,xSize=5, ySize=7):
    fig0, ax0 = plt.subplots(xSize, ySize, figsize=(15,6))
    fig0.subplots_adjust(hspace=0.2, wspace=0.1)
    ax0 = ax0.ravel()
    
    dct = indexClass(dataLabel)
    c0, c1   = 0, 0
    img0 = np.zeros([32,32,3],dtype=np.uint8)
    img0.fill(255)
    
    for i in range(xSize*ySize):
        if i in range(0,xSize*ySize,ySize):
            # myImg
            image = myImg[c0].squeeze()
            title = myLabel[c0]
            ax0[i].set_title(title, fontsize=8)
            c0 += 1
        elif i in range(1,xSize*ySize,ySize):
            # blank
            image = img0[:]
            title= '' 
            ax0[i].set_title(title, fontsize=8)
        else:
            # dataImg
            idCls = top_K.indices[c0-1][c1%(ySize-2)]
            title = top_K.values[c0-1][c1%(ySize-2)]*100
            title = title.astype(int)
            ax0[i].set_title(str(idCls)+':'+str(title)+'%', fontsize=8)
            c1 += 1
            index = random.randint(dct[idCls][0], dct[idCls][-1])
            image = dataImg[index].squeeze()
        
        cMap, ch = chMap(image)
        
        if ch == 3:
            ax0[i].imshow(image)
        else:
            ax0[i].imshow(image, cmap = cMap)
        ax0[i].axis('off')


def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    ### Visualize your network's feature maps here.
    # image_input: the test image being fed into the network to produce the feature maps
    # tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
    # activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
    # plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")




def main():
    args = parse_args()

    ch = int(input('< RGB > images : 3 OR < GRAY > images : 1 = '))
    if ch == 3:
        tK = topKPrediction(myImg1, pathCkp, 43)
    else:
        tK = topKPrediction(myImg2, pathCkp, 43)
    print('Top k Softmax Probabilities : {}'.format(tK))


    # 
    fig, ax0 = plt.subplots(5, 2, figsize=(20, 10))
    fig.subplots_adjust(hspace = 0.2, wspace=0.1)
    ax0 = ax0.ravel()
    for i, classId, value, image, label in zip(range(0,10,2), tK.indices, tK.values, myImage, myLabel):
        ax0[i].set_title(label, fontsize=8)
        ax0[i].axis('off')    
        ax0[i].imshow(image)
        ax0[i+1].yaxis.grid(color='#eeeeee')
        ax0[i+1].bar(classId, value, color='#616161')


    # 
    showTrace3(X_train, y_train, myImg1, myLabel, tK)

    # 
    showTrace3(X_train, y_train, myImg2, myLabel, tK ,xSize=5, ySize=35)


if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''