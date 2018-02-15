from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
from utils_step2 import dataVisu, showTrace
import pickle
import numpy as np
from numpy import newaxis
import cv2
import matplotlib.pyplot as plt
import os

# Helper function: un-pickle python object 
class hFct(object):
    '''Helper functions'''
    def __init__(self,path,name,pyObj):
        self.path  = path
        self.name  = name
        self.pyObj = pyObj

    # Save python objects 
    def serialize(self):
        """Pickle a Python object"""      
        with open(self.path+self.name, "wb") as pfile:
            pickle.dump(self.pyObj, pfile)

    # Load python objects 
    def deserialize(self):
        """Extracts a pickled Python object and returns it"""
        with open(self.path+self.name, "rb") as pfile:
            dataSet = pickle.load(pfile)
        return dataSet

    # Load pickled data 
    def loadValid(self):       
        dataSet = self.deserialize()
        dataImg, dataLabel  = dataSet['features'], dataSet['labels']
        try:
            dataSize, dataCoord = dataSet['sizes'], dataSet['coords']
        except:
            dataSize, dataCoord = {}, {}
        finally:
            return dataImg, dataLabel, dataSize, dataCoord  


# Helper function: preprocess the data - part 1
class dataPPro(object):
    def __init__(self, args, dataImg, dataLabel, dataSize, dataCoord, path=''):
        self.args      = args
        self.dataImg   = dataImg
        self.dataLabel = dataLabel
        self.s         = dataSize
        self.c         = dataCoord
        self.path      = path

      
    def getImg(self):
        return self.dataImg
        
    def setImg(self, newImg):
        self.dataImg = newImg.copy()
        
    def getLabel(self):
        return self.dataImg
        
    def setLabel(self, newLabel):
        self.dataLabel = newLabel.copy()
        
      
    # Preprocess the data: RGB > grayscale
    def proGray(self):
        outImg = np.empty_like(self.dataImg)*0
        outImg = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.dataImg ])
        #outImg = outImg[..., newaxis]
        self.setImg(outImg)
        return outImg, self.dataLabel, self.s, self.c


    # Preprocess the data: input{RGB,GRAY} > sharpen
    def proShp(self):
        kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
        outImg = np.array([cv2.filter2D(img, -1, kernel) for img in self.dataImg])
        #outImg = outImg[..., newaxis]
        self.setImg(outImg)
        return outImg, self.dataLabel, self.s, self.c
    
    
    # Preprocess the data: sharpen > equalized histogram
    def proHst(self):
        outImg = np.array([cv2.equalizeHist(img) for img in self.dataImg ])
        #outImg = outImg[..., newaxis]
        self.setImg(outImg)
        return outImg, self.dataLabel, self.s, self.c
    

    # Preprocess the data: equalized histogram > CLAHE
    def proClahe(self):
        '''CLAHE - Equalize (adaptively with limited contrast) the histogram of a globaly equalize image'''
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        outImg = np.array([clahe.apply(img) for img in self.dataImg ])
        #outImg = outImg[..., newaxis]
        self.setImg(outImg)
        return outImg, self.dataLabel, self.s, self.c

    
    # Preprocess the data: CLAHE > center & normalize images
    def proCtrNrm(self):
        '''Source: SDCNP, Lesson 8, lecture 23-Normalized Inputs and Initial Weights'''
        outImg = (self.dataImg - np.mean(self.dataImg))/np.std(self.dataImg) # zero-center&normalize
        return outImg, self.dataLabel, self.s, self.c
              
    
    def __str__(self):
        return '< dataPPro >' 

# Helper function: preprocess the data - part 2
def proAll(args, ltName, ltData, ltSzcr, path, ltMeth, ltSufx):
    '''
    ltName =['train','valid','test']
    ltData =[(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]
    ltSzcr =[(s_train, c_train), (s_valid, c_valid), (s_test, c_test)]
    path   = pathPro
    ltMeth =[dataPPro.proGray, dataPPro.proShp, dataPPro.proHst, dataPPro.proClahe]
    ltSufx =['1Gray', '2Shp', '3Hst', '4Clahe']
    '''
    
    dtPrOut = {}    
    
    for i0, i1, i5 in zip(ltName, ltData, ltSzcr):
        dtPrIn = dataPPro(args, i1[0], i1[1], i5[0], i5[1])
        
        flag = None

        for i2, i3 in zip(ltMeth, ltSufx):
            if flag is None: # Center and normalize the RGB images
                dtPrOut['features'], dtPrOut['labels'], dtPrOut['sizes'], dtPrOut['coords'] = dataPPro.proCtrNrm(dtPrIn)
                hFct(path, i0+'_0Rgb.p', dtPrOut).serialize()   # serialize(dtPrOut, path, i0+'_0Rgb.p')
                dtPrOut = {}
                flag = False                    
                    
            dtPrOut['features'], dtPrOut['labels'], dtPrOut['sizes'], dtPrOut['coords'] = i2(dtPrIn)         
            dtPrOut['features'], dtPrOut['labels'], dtPrOut['sizes'], dtPrOut['coords'] = dataPPro.proCtrNrm(dtPrIn)
          
            if dtPrOut['features'].shape[-1] == 32:
                dtPrOut['features'] = dtPrOut['features'][..., newaxis]               
                
            hFct(path, i0+'_'+i3+'.p', dtPrOut).serialize()   # serialize(dtPrOut, path, i0+'_'+i3+'.p')
            dtPrOut = {}       
        del dtPrIn

# Helper function: preprocess the data - part 3
def creaShow(pathIn):
    '''
    pathIn=[pathData, pathPro]
    '''
    cMap = ['rgb', 'gray']
    for path, cmap in zip(pathIn, cMap):
        for infile in os.listdir(path):
            if infile[-2:] == '.p' and '5CtrNrm' not in infile:
                try: # Download python objects
                    dat0Img, dat0Label, dat0Size, dat0Coord = hFct(path, infile, '').loadValid()
                except:
                    dat0Img, dat0Label  = hFct(path, infile, '').loadValid()
                    dat0Size, dat0Coord = {}, {}
                # Show a few sample of the 
                showTrace(dat0Img,infile,xSize=1, ySize=8)


def main():
    args = parse_args()

    # load the dataset
    X_train, y_train, s_train, c_train = data_load(args, 'train.p')
    X_valid, y_valid, s_valid, c_valid = data_load(args, 'valid.p')
    X_test, y_test, s_test , c_test    = data_load(args, 'test.p')

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