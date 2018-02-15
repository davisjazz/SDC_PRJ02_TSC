from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
from utils_step1 import ocrLabel, indexClass, dataExplo
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import gridspec
import PIL
from PIL import Image

# Helper function: show43TS, show5TS, show5TS2, showMosaic, makeSprite, showSprite
class dataVisu(object):
    def __init__(self, args, dataImg, dataLabel, imgSize, bBox): # self.args.image
        self.args      = args
        self.dataImg   = dataImg
        self.dataLabel = dataLabel
        self.sprImg    = None
        self.s         = imgSize
        self.c         = bBox
        

    def setImg(self, newImg):
        self.dataImg   = newImg    
        
    def getTab(self):
        return self.args.tab
        
    def setTab(self, newTab):
        self.args.tab=newTab
        assert len(self.args.tab) == 2, '[Error] your tab should be: Tab = [nbr of row, nbr col]'        
            
                
    def show43TS(self, figNb=4):
        '''Output: Show 43 random images, one per class, with the bounding box around the sign in the image
        '''
        # Parameters
        n_classes = len(np.unique(self.dataLabel))
        xDim, yDim = self.args.tab[0],self.args.tab[1]
        sIze = self.dataImg.shape[1]

        # Get dictionary[idClass] = tuple( index(idClass))
        dct, dct0, dct1, lt = {}, {}, {}, []
        dct0, dct1, lt = ocrLabel(self.args, self.dataLabel)
        dct = indexClass(self.dataLabel)

        # ...
        fig0, ax0 = plt.subplots(xDim, yDim, figsize=(15,6))
        fig0.subplots_adjust(hspace=0.2, wspace=0.1) #hspace=0.2, wspace=0.1) # hspace=0.05, wspace=0.05
        ax0 = ax0.ravel()

        # Print title
        tiTle = '43 random images, one per class'
        print()
        print('fig.'+str(figNb)+': '+tiTle)

        # Show 43 traffic sign, one image per class
        count = 0
        for i in range(xDim*yDim):
            rx    = sIze/self.s[i][0]
            ry    = sIze/self.s[i][1]
            if count < n_classes:
                index = random.randint(dct[count][0], dct[count][-1]) 
                image = self.dataImg[index]
                
                bbox  = patches.Rectangle((self.c[i][0]*rx, self.c[i][1]*ry), (self.c[i][2]-self.c[i][0])*rx, (self.c[i][3]-self.c[i][1])*ry, edgecolor='#ffffcc',facecolor='none')
                ax0[i].add_patch(bbox)
                
                if image.shape[-1]==1:
                    ax0[i].imshow(image, cmap = 'gray')
                elif image.shape[-1]==3:
                    ax0[i].imshow(image)           
                ax0[i].axis('off')
                tl0 = lt[count][1][:17]+'.'
                ax0[i].set_title(tl0, fontsize=8)
            else:
                ax0[i].axis('off')
            count += 1
        fig0.savefig(self.args.image+'_dataVisu_4_show43TS.png')


    def show5TS(self, figNb=5):
        # Parameters
        n_classes = len(np.unique(self.dataLabel))
        xDim, yDim = self.args.tab[0],self.args.tab[1]

        # Get dictionary[idClass] = tuple( index(idClass))
        dct, dct0, dct1, lt = {}, {}, {}, []
        dct0, dct1, lt = ocrLabel(self.args,self.dataLabel)
        dct = indexClass(self.dataLabel)

        # ...
        fig0, ax0 = plt.subplots(xDim, yDim, figsize=(15,6))
        fig0.subplots_adjust(hspace=0.2, wspace=0.1) #hspace=0.2, wspace=0.1) # hspace=0.05, wspace=0.05
        ax0 = ax0.ravel()

        # Print title
        tiTle = 'several images of the same traffic signs'
        print()
        print('fig.'+str(figNb)+': '+tiTle)

        # Show 43 traffic sign, one image per class
        count = random.randint(0, n_classes-1)
        itr   = 0
        for j in range(xDim):
            for i in range(yDim):
                dctLen = len(dct[count])
                indEx  = random.randint(0,dctLen-1)
                index  = dct[count][indEx]
                image = self.dataImg[index]
                ax0[itr].imshow(image, cmap=self.args.cmap)
                ax0[itr].axis('off')
                title = lt[count][1][:17]+'.'
                ax0[itr].set_title(title, fontsize=8)
                itr += 1
            tmp = count
            while tmp == count:
                count = random.randint(0, n_classes-1)
        fig0.savefig(self.args.image+'_dataVisu_5_show5TS.png')
        
        
                
    def show5TS2(self, figNb=5):
        # Parameters
        n_classes = len(np.unique(self.dataLabel))
        xDim, yDim = self.args.tab[0],self.args.tab[1] # tab[5, 10] <- default values
        sIze = self.dataImg.shape[1]         # <-- show43TS

        # Get dictionary[idClass] = tuple( index(idClass))
        dct, dct0, dct1, lt, lt0 = {}, {}, {}, [], []
        dct0, dct1, lt = ocrLabel(self.args, self.dataLabel)
        dct = indexClass(self.dataLabel)
        lt0 =['dataPPro.proGray', 'dataPPro.proShp', 'dataPPro.proHst', 'dataPPro.proClahe', 'dataPPro.proCtrNrm']

        # ...
        fig0, ax0 = plt.subplots(xDim, yDim, figsize=(15,6))
        fig0.subplots_adjust(hspace=0.2, wspace=0.1) #hspace=0.2, wspace=0.1) # hspace=0.05, wspace=0.05
        ax0 = ax0.ravel()

        # Print title
        count = random.randint(0, n_classes-1) #title = lt[count][1][:17]+'.'
        tiTle = 'same traffic signs after being preprocceed | TS name = '+lt[count][1]+'.'
        print()
        print('fig.'+str(figNb)+': '+tiTle)

        # Show traffic sign: 5 lines, Z columns
        count = random.randint(0, n_classes-1)
        itr   = 0
        for j in range(xDim): # xDim = 5
            for i in range(yDim): # yDim = 10
                dctLen = len(dct[count])
                indEx  = random.randint(0,dctLen-1)
                index  = dct[count][indEx]
                image = self.dataImg[index]
                ax0[itr].imshow(image, cmap=self.args.cmap)
                ax0[itr].axis('off')
                title =  lt0[itr%yDim] #title = lt[count][1][:17]+'.'
                ax0[itr].set_title(title, fontsize=8)
                itr += 1
            tmp = count
            while tmp == count:
                count = random.randint(0, n_classes-1) 
        fig0.savefig(self.args.image+'_dataVisu_5_show5TS2.png')
            
            
    def showMosaic(self, figNb=6):
        # Number of classes/labels there are in the dataset
        n_classes = len(np.unique(self.dataLabel))

        # Print title
        tiTle = 'mosaic consists of 43 x 5 images, one patern per class'
        print()
        print('fig.'+str(figNb)+': '+tiTle)
        
        # Get dictionary[idClass] = tuple( index(idClass))
        dct, dct0, dct1, lt = {}, {}, {}, []
        dct0, dct1, lt = ocrLabel(self.args, self.dataLabel)
        dct = indexClass(self.dataLabel)
        
        # Compute the width and the height of the sprite image
        dIm = n_classes**0.5
        dIm = int(dIm) + (dIm - int(dIm) > 0)
        master_width, master_height = dIm*3, dIm*3
        gs1 = gridspec.GridSpec(master_width, master_height)
        gs1.update(left=0, right=0.5, hspace=0.2, wspace=0.1) #hspace=0.2, wspace=0.1) # hspace=0.05, wspace=0.05# hspace=0.05, wspace=0.05)
               
        # Create and save the sprite image
        count = 0
        for y0 in range(dIm):
            y = y0*3
            for x0 in range(dIm):
                x = x0*3
                lt1 = [ [0+x,y+0], [0+x,y+2], [1+x,y+2], [2+x,y+2], [2+x,y+1], [2+x,y+0] ]
                flag = True
                
                if count+1 <= n_classes:
                    for i in lt1:
                        if flag:
                            ax = plt.subplot(gs1[0+x:x+2,0+y:y+2])            
                            flag = False
                        else:
                            ax = plt.subplot(gs1[i[0],i[1]])
                        index = random.randint(dct[count][0], dct[count][-1]) 
                        image = self.dataImg[index]
                        ax.imshow(image, cmap=self.args.cmap)
                        ax.set_xticks([]); ax.set_yticks([])
                else:
                    ax = plt.subplot(gs1[x,y])
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.axis('off')            
                    
                count += 1
        #fig0.savefig(self.args.image+'_dataVisu_6_showMosaic.png')
            
    
    def makeSprite(self, cMap='gray', fileName='xx', tRace1=False, tRace2=False):
        '''Input : cMap = 'rgb' or  'gray'
           Output: a single image, called a sprite image, that is a collection of all images contained in the dataImg
           Note  : it will be useful when we implement the Embedding Visualizer in TensorBoard (tensorflow)
        '''
        if tRace2: print('[{}] {} : INPUT | dataImg.shape = {}'.format('dataVisu','makeSprite',self.dataImg.shape))
        ## Create a blank sprite image
        #-1/ Import libraries
        #import PIL
        #from PIL import Image

        # 0/ Retrieve the name of self.dataImg
        outputName = '_sp_'+fileName
        lt=[]
        if tRace2:print('[{}] {} : {} = {}'.format('dataVisu','makeSprite', 'Retrieve the name of self.dataImg',outputName ))

        #1-Retrieve the width and the height of image input
        image_width, image_height = self.dataImg.shape[1], self.dataImg.shape[2]

        #2-Compute the width and the height of the sprite image
        dIm = len(self.dataImg)**0.5
        dIm = int(dIm) + (dIm - int(dIm) > 0)
        pxSprite = dIm*image_width
        self.sprImg_width, self.sprImg_height = pxSprite, pxSprite
        if tRace2: print('[{}] {} : sprImg_width = {}, sprImg_height = {}'.format('dataVisu','makeSprite',self.sprImg_width,self.sprImg_height))


        outputName = outputName+'_{}x{}'.format(self.sprImg_width, self.sprImg_height)
        if tRace2: print('[{}] {} : sprite image name = {}.{}'.format('dataVisu','makeSprite',outputName,self.args.png))

        #3-Create a blank sprite image
        self.sprImg = Image.new(mode='RGBA', size=(self.sprImg_width, self.sprImg_height), color=None)
        if tRace2: print('[{}] {} : The blank sprite image is {}x{}'.format('dataVisu','makeSprite',self.sprImg_width, self.sprImg_height))

        ## Create and save the sprite image
        count = 0
        for y in range(dIm):
            for x in range(dIm):
                try:
                    if cMap == 'rgb' : image  = Image.fromarray(self.dataImg[count],'RGB')
                    if cMap == 'gray': image  = Image.fromarray(self.dataImg[count],'L')
                    lt.append(self.dataLabel[count])
                except:
                    if cMap == 'rgb' : image  = Image.new(mode='RGB', size=(self.sprImg_width, self.sprImg_height), color=0)
                    if cMap == 'gray': image  = Image.new(mode='L'  , size=(self.sprImg_width, self.sprImg_height), color=None)
                    lt.append(99)
                finally:                
                    if tRace1: print('[{}] {} : adding image {}'.format('dataVisu','makeSprite',count+1),' at location {}'.format((x,y)))
                    if tRace2 and ((count+1)%int(dIm*dIm/10)==0): print('[{}] {} : adding image {}'.format('dataVisu','makeSprite',count+1),' at location {}'.format((x,y)))
                    
                    self.sprImg.paste(image,(x*image_width,y*image_height))
                    image.close()
                    count+= 1

        sprit_name = self.args.image+outputName+'.'+self.args.png
        self.sprImg.save(sprit_name) #, transparency=0 )
        if tRace2: print('[{}] {} : {}.{} has been created and saved'.format('dataVisu','makeSprite',outputName, self.args.png))

        spriteLbl = self.args.image+outputName+'.tsv'
        np.savetxt(spriteLbl, lt, '%1i')
        if tRace2: print('[{}] {} : {}.{} has been created and saved'.format('dataVisu','makeSprite',outputName, '.tsv'))

        return sprit_name, self.sprImg
    
    
    def showSprite(self, figNb=7):
        '''It shows the sprite image into the jupyter notebook
        '''
        from matplotlib.pyplot import imshow
        import numpy as np
        from PIL import Image

        # Print title
        tiTle = 'a single image of all images contained in the data set'
        print()
        print('fig.'+str(figNb)+': '+tiTle)
        
        get_ipython().run_line_magic('matplotlib', 'inline')
        try:
            imshow(np.asarray(self.sprImg))
        except:
            imshow(self.sprImg)
        imshow(self.sprImg)
                        
                        
    def __str__(self): # footnote [3]
        return '< dataVisu >'


# Helper function: Show xSize*ySize images
def showTrace(dataImg,title='',xSize=1, ySize=8):
    fig0, ax0 = plt.subplots(xSize, ySize, figsize=(15,6))
    fig0.subplots_adjust(hspace=0.2, wspace=0.1)
    ax0 = ax0.ravel()

    for i in range(xSize*ySize): 
        image = dataImg[i].squeeze()
        #print('[INPUT]image.shape: {}'.format(image.shape))
        
        ch = len(image.shape)
        #print('[INPUT]ch = len dataImg.shape: {}'.format(ch))
        
        if image.shape[-1] == 3:        
            cMap='rgb'
            ax0[i].imshow(image)
        elif image.shape[-1] == 32 or image.shape[-1] == 1:
            cMap='gray'
            ax0[i].imshow(image, cmap = cMap)
        else:
            raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

        #ax0[i].imshow(image, cmap = cMap)
        ax0[i].set_title(title, fontsize=8)
        ax0[i].axis('off')


# Helper function: Show xSize*ySize images
def showTrace2(dataImg, dataLabel, xSize=1, ySize=5):
    fig0, ax0 = plt.subplots(xSize, ySize, figsize=(15,6))
    fig0.subplots_adjust(hspace=0.2, wspace=0.1)
    ax0 = ax0.ravel()

    # Get dictionary[idClass] = tuple( index(idClass))
    dct, dct0, dct1, lt = {}, {}, {}, []
    dct0, dct1, lt = ocrLabel(dataLabel)

    for i in range(xSize*ySize): 
        image = dataImg[i].squeeze()
        
        if image.shape[-1] == 3: 
            ch = 3
            cMap='rgb'
            ax0[i].imshow(image)
        elif image.shape[-1] == 32 or image.shape[-1] == 1:
            ch = 1
            cMap='gray'
            ax0[i].imshow(image, cmap = cMap)
        else:
            raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

        #ax0[i].imshow(image, cmap = cMap)
        title = dct0[dataLabel[i]][:17]+'.'
        ax0[i].set_title(title, fontsize=8)
        ax0[i].axis('off')


def main():
    args = parse_args()

    # load the dataset
    X_train, y_train, s_train, c_train = data_load(args, 'train.p')
    X_valid, y_valid, s_valid, c_valid = data_load(args, 'valid.p')
    X_test, y_test, s_test , c_test    = data_load(args, 'test.p')


if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''