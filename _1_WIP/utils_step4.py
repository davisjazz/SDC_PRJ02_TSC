from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
from utils_step2 import dataVisu, showTrace
from utils_step3 import hFct, dataPPro, proAll, creaShow
import numpy as np
from numpy import newaxis
import pickle
import csv, cv2
import os
import random
import prettyplotlib as ppl
import brewer2mpl
import matplotlib.pyplot as plt

# Helper function: perturbation in position
def jitShift(dataImg, pXl=2):
    '''Image is randomly perturbed in position ([-pXl,pXl] pixels)
       Output: the image is translated random values between -pXl and pXl
    ''' 
    xPxl, yPxl, count = 0, 0, 0
    outImg = np.empty_like(dataImg)*0    

    while xPxl==0 and yPxl==0:
        xPxl = random.randint(-pXl, pXl)
        yPxl = random.randint(-pXl, pXl)
        count += 1
        if count > 3:
            xPxl = yPxl = pXl
            break
    rows, cols = dataImg.shape[0], dataImg.shape[1]
    M = np.float32([[1,0,xPxl],[0,1,yPxl]])
    outImg = cv2.warpAffine(dataImg,M,(cols,rows))
    return outImg


# Helper function: perturbation in rotation
def jitRot(dataImg, theta1=7, theta2=15):
    '''Rotation of an image for an angle θ is achieved by the transformation matrix of the form
       Output: the images are rotated values between -theta and theta
    '''
    rot, rot1, rot2 = 0, 0, 0   
    outImg = np.empty_like(dataImg)*0    

    rot1 = np.random.uniform(-theta2, -theta1) #random.randint(-theta, theta)
    rot2 = np.random.uniform(theta1, theta2) 
    rot  = random.choice([rot1,rot2])
    
    rows, cols = dataImg.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
    outImg = cv2.warpAffine(dataImg,M,(cols,rows))

    return outImg


# Helper function: bounding box to crop the image
def jitCrop(dataImg, s, c, zoomIn=True, pad=2):
    '''Use the bounding box to crop the image, then resize it to get 32x32 images
    '''
    # x = 5
    try:
        a = s[0]/s[1]
        b = s[1]/s[0]
    except:
        raise ZeroDivisionError('s=',s)
    finally:
        if a>2 or b>2:
            return dataImg
            
        else:
            # create empty numpy array
            rows, cols = dataImg.shape[:2]          
            # use the bounding box to crop the image
            rx = rows/s[0]
            ry = cols/s[1]          
            if zoomIn:
                nC = [ int(c[0]*rx + pad), int(c[1]*ry - pad), int(c[2]*rx + pad), int(c[3]*ry - pad) ]
            else:
                nC = [ int(c[0]*rx - pad), int(c[1]*ry + pad), int(c[2]*rx - pad), int(c[3]*ry + pad) ]
         
            if min(nC) < 0: # http://stackoverflow.com/questions/11764260/how-to-find-the-minimum-value-in-a-numpy-matrix
                nC = [ int(c[0]*rx), int(c[1]*ry), int(c[2]*rx), int(c[3]*ry) ]
       
            outImg = np.empty_like(dataImg[ nC[1]:nC[3], nC[0]:nC[2] ])*0
            outImg = dataImg[ nC[1]:nC[3], nC[0]:nC[2] ]
                 
            # resize the cropped images to get 32x32 images
            outImg = cv2.resize(outImg,(cols,rows),interpolation = cv2.INTER_CUBIC) # cv2.INTER_LINEAR)
            return outImg  

# Helper function: progress bar
def barPrg(args,i0,i1,dataLabel,ptTsn=False, incr0=500,str0='-'):
    ''' Progress bar for jittered data
    Input: {i0: index of the traffic sign,\
            i1: number of generated data,\
            prtTsn: 'yes' if you want to print the TS name}    
    '''
    dct0, dct1, lt = ocrLabel(args, dataLabel)
    if ptTsn:
        print('{:30}: '.format(lt[i0][1][:30]), end='')
    else:
        if i1 % incr0 == 0:
            if i1 == 0:
                print('', end='')
            else:
                print('|', end='')
        elif i1 % int(incr0/5) == 0:
            print(str0+'',end='')


# Helper function: generate fake data - part 1
def jitData(args, dataImg, dataLabel, size, coord, qty=100):
    # Initialization
    copyImg   = dataImg.copy()
    copyLabel = dataLabel.copy()
    n_classes = len(np.unique(dataLabel))
    lt = []

    for i0 in range(n_classes):
        classIndex = np.where(copyLabel == i0)
        nSamples   = len(classIndex[0])
        delta      = qty - nSamples

        barPrg(args,i0,0,dataLabel,True) # initiate the progress bar
        
        if nSamples < qty and nSamples!= 0:
            
            outImg   = np.empty_like(dataImg[:delta])*0
            outLabel = np.empty_like(dataLabel[:delta])*0 
            
            for i1 in range(delta):
                index = classIndex[0][i1 % nSamples]
                if i1%2 == 0:
                    #image = jitCrop(jitRot(jitShift(dataImg[index])), size[index], coord[index], False)
                    image = jitRot(jitShift(dataImg[index]))
                else:
                    #image = jitCrop(jitRot(jitShift(dataImg[index])), size[index], coord[index], True)
                    image = jitRot(jitShift(dataImg[index]))

                outLabel[i1] = i0
                
                if image.shape[-1] == 3 or image.shape[-1] == 1:
                    outImg  [i1] = np.float32(image)
                elif image.shape[-1] == 32:
                    outImg  [i1] = np.float32(image[..., newaxis])
                                
                image = np.empty_like(image)*0
                
                barPrg(args,i0,i1,dataLabel,False) # show the progression of the process

            copyImg   = np.float32( np.concatenate((copyImg  , outImg)) )
            copyLabel = np.concatenate((copyLabel, outLabel))
            lt.append(len(outImg))
            
        print('') 
            
    print()
    print('Legend: [ ''-'' = 100 ] , [ ''|'' = 500 ] ')
    print()
    print('[{:6}] : ''('' {:10} , {:10} '')'' = ''('' {:4}, {:4}'')'''.format('BEFORE','X_train'   , 'y_train'   , len(dataImg), len(dataLabel)))
    print('[{:6}] : ''('' {:10} , {:10} '')'' = ''('' {:4}, {:4}'')'''.format('AFTER' ,'X_trainJIT', 'y_trainJIT', len(copyImg), len(copyLabel)))
    print('[{:6}] : ''('' {:10} '')'' = ''('' {:4} '')'''.format('AFTER' ,'qty data',sum(lt)))

    return copyImg, copyLabel, outImg, outLabel


# Helper function: generate fake data - part 2
def jitItall(args, pathIn, pathOut, qty=500):
    '''Generate jittered data for each type of preproceeded data
    pathIn =[pathData, pathPro]
    pathOut=pathJit
    '''
    #cMap = ['rgb', 'gray']
    lt1, lt2 = [], []
    for file1,file2 in zip(os.listdir(pathOut+'full/'),os.listdir(pathOut+'addon/')):
        lt1.append(file1)
        lt2.append(file2)
    
    for path in pathIn:
        for infile in os.listdir(path):
            
            infileFull  = 'JIT_full_'+str(qty)+'_'+infile
            infileAddon = 'JIT_addon_'+str(qty)+'_'+infile
            
            if infileFull not in lt1 or infileAddon not in lt2:            
                if infile[-2:] == '.p' and 'train' in infile:      
                #if infile[-2:] == '.p':          
                    # Download python objects
                    dat0Img, dat0Label, dat0Size, dat0Coord = hFct(path, infile, '').loadValid()      
                    # Generate fake data
                    dat1Img, dat1Label, dat2Img, dat2Label = jitData(args, dat0Img, dat0Label, dat0Size, dat0Coord, qty)   
                    # Save python objects
                    if infileFull not in lt1:
                        dtPrOut1 = {}
                        dtPrOut1['features'] = dat1Img
                        dtPrOut1['labels']   = dat1Label
                        hFct(pathOut+'full/', infileFull, dtPrOut1).serialize()   # serialize(dtPrOut1, pathOut, infileFull)
                    if infileAddon not in lt2:
                        dtPrOut2 = {}
                        dtPrOut2['features'] = dat2Img
                        dtPrOut2['labels']   = dat2Label
                        hFct(pathOut+'addon/', infileAddon, dtPrOut2).serialize()   # serialize(dtPrOut2, pathOut, infileAddon)


# Helper function: show the list and histogram of jittered data
def jitListChart(args, path, infile='JIT_full_500_train.p', oPtion='111'):
    '''
    path=pathFull
    '''
    try:
        dataImg, dataLabel = hFct(path, infile, '').loadValid() # loadValid(path, infile)
    except:
        dataImg, dataLabel, dataSize, dataCoord = hFct(path, infile, '').loadValid()
    jitTrainLb_explo = dataExplo(args, dataLabel)
    if oPtion[0]=='1':
        jitTrainLb_explo.showList(1)
    if oPtion[1]=='1':
        jitTrainLb_explo.showChart(2)
    if oPtion[2]=='1':
        jitTrainLb_explo.showDist(3)


def main():
    args = parse_args()

    # load the dataset
    X_train, y_train, s_train, c_train = data_load(args, 'train.p')
    X_valid, y_valid, s_valid, c_valid = data_load(args, 'valid.p')
    X_test, y_test, s_test , c_test    = data_load(args, 'test.p')

    # [test] wip - 05 - generate additional data
    ## OK - generate fake data up to get 500 to 3000 occurences per traffic signs
    for i in range(500,1000,500): # 3001,500):
        jitItall(args, [args.dtset, args.ppro], args.jit, i)
    ## Ok - show the original image, the shifted version, the rotated version, and the combinaison of the two perturbations 
    path, infile, a = args.dtset, 'test.p', 6
    dataImg, dataLabel, dataSize, dataCoord = hFct(path, infile, '').loadValid()
    dataLabel, dataSize, dataCoord = None, None, None
    dtImg1, dtImg2, dtImg3, dtImg4 = dataImg[a], jitShift(dataImg[a], pXl=5), jitRot(dataImg[a]), jitRot(jitShift(dataImg[a], pXl=5))
    dtImg0 = np.concatenate(([dtImg1], [dtImg2], [dtImg3], [dtImg4]))
    showTrace(dtImg0,title='',xSize=1, ySize=4)
    ## OK - show images of jittered data
    creaShow([args.full])
    ## OK - show the list and histogram of jittered data
    jitListChart(args, args.full,'JIT_full_500_train.p','010')

    # [test] OK - 04 - proAll, creaShow
    ## preprocess the data: preprocessed and save all in pathPro
    proAll(args,
           ['train','valid','test'],
           [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)],
           [(s_train, c_train), (s_valid, c_valid), (s_test, c_test)],
           args.ppro,
           [dataPPro.proGray, dataPPro.proShp, dataPPro.proHst, dataPPro.proClahe],
           ['1Gray', '2Shp', '3Hst', '4Clahe'])
    ## visualize the pre-processed Data
    creaShow([args.dtset,args.ppro])  #  [pathData, pathPro])

    # [test] OK - 03 - dataVisu.show43TS,dataVisu.show5TS,dataVisu.showMosaic,dataVisu.makeSprite
    #visuTrain = dataVisu(args, X_train, y_train, s_train, c_train)  # path = args.tboard
    #visuTrain.show43TS()
    #visuTrain.show5TS()
    #visuTrain.showMosaic()
    #t1,t2= visuTrain.makeSprite('rgb')

    # [test] OK - 02 - dataExplo.showList, .showChart, .showDist
    #label = dataExplo(args, y_train)
    #label.showList(1)
    #label.showChart(2)
    #label.showDist(3)

    # [test] OK - 01 - ocrLabel(), indexClass()
    #dct0, dct1, lt = ocrLabel(args, y_valid)
    #dct0 = indexClass(y_valid)

    #print(' : {}'.format())

if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''