from utils_step0 import parse_args, dir_check, dir_create, data_load, chMap
import numpy as np
import csv
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import brewer2mpl

# Helper function: map class id, occurence, Traffic-sign names
def ocrLabel(args, dataLabel):
    n_classes = len(np.unique(dataLabel))
    # Map class_id with Traffic-sign names. Output : dct[classId] = Traffic-sign names
    with open(args.file_csv, newline='', encoding="utf8") as file_csv1:
        read0 = csv.reader(file_csv1)
        dct0, dct1 = {}, {}
        for i in read0:
            try:
                dct0[int(i[0])] = i[1]
            except:
                pass
    # Occurence by class id. Output : dct[classId] = occurence
    ocr, classId = np.histogram(dataLabel, np.arange(n_classes+1))
    classId = classId[:-1].copy()
    for i,j in zip(classId,ocr):
            dct1[i] = j
    # Occurence by Traffic-sign names. Output : lt[classId] = [occurence, Traffic-sign names] 
    lt = []
    for i in classId:
            lt.append([dct1[i], dct0[i]])
    return dct0, dct1, lt

# Helper function: dict[class id] = 1D array of all related indexes
def indexClass(dataLabel):
    '''Output: a dictionary that has 43 keys (= class id)
               and its values are an 1D array containing all the indexes of a each class
    '''
    n_classes = len(np.unique(dataLabel))
    dct = {}
    tl  = ()
    for i in range(n_classes):
        tl     = np.where(dataLabel == i) # tuple = array of Index(class)
        dct[i] = tl[0] # dictionary[key=idClass] = array of Index(idClass)
    return dct


# Helper function: showChart, showDistribution, showList
class dataExplo(object):
    def __init__(self, args, dataLabel, sOrted=False):
        assert dataLabel.ndim == 1, 'Please, consider use the 1D array containing the label/class id of the traffic sign'
        self.args      = args
        self.dataLabel = dataLabel
        self.sOrted    = sOrted
    

    def showChart(self,figNb=2):
        '''Show the number of occurence per class'''
        n_classes = len(np.unique(self.dataLabel))
        set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors # [2] Get "Set2" colors from ColorBrewer
        hist, bins = np.histogram(self.dataLabel, np.arange(n_classes+1))
        fig = plt.figure(figsize=(16, 3))

        plt.title('fig.'+str(figNb)+': labels - number of occurence per class')
        plt.xlabel('Class id')
        plt.ylabel('Number of occurence')
        plt.xlim(0,n_classes)
        plt.ylim(0,np.amax(hist))
        ax = fig.add_subplot(111)
        ppl.bar(ax, bins[:-1], hist, grid='y', color='#616161')
        
        # Save the result into file:
        fig.savefig(self.args.image+'_dataExplo_2_showChart.png')
     

    def showDist(self,figNb=3):
        n_classes = len(np.unique(self.dataLabel))
        set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
        axX = np.arange(len(self.dataLabel))
        axY = self.dataLabel
        fig = plt.figure(figsize=(16,2))

        plt.title('fig.'+str(figNb)+': labels - data distribution')
        plt.xlabel('Index')
        plt.ylabel('Class id')
        plt.ylim(0,n_classes)
        ax = fig.add_subplot(111)    
        plt.plot(axX,axY,".", color=set2[7]) #ax.scatter(axX, axY, color=set2[7])
        # Save the result into file:
        fig.savefig(self.args.image+'_dataExplo_3_showDist.png')
          

    def showList(self,figNb=1): # _dataExplo_1_showList.png
        dt1 = { 0: '----------------------------------------------',                1: '{:47}',                2: '{:47}',                3: '{:40}',                4: '|{:4}'                }

        dt2 = {10: 'Traffic sign name',               11: 'Qty',               }


        if self.sOrted:
            sTr = 'sorted'
        else:
            sTr = ''
        print('fig.'+str(figNb)+': labels - {} List of occurence per Traffic sign name'.format(sTr))
        
        # Print the table header
        print(dt1[1].format(dt1[0]), dt1[1].format(dt1[0]))
        print(dt1[3].format(dt2[10]), dt1[4].format(dt2[11])              +'  '+dt1[3].format(dt2[10]), dt1[4].format(dt2[11]))
        print(dt1[1].format(dt1[0]), dt1[1].format(dt1[0]))

        # Print ( the Traffic sign name and the related occurence ) x 2 / line
        dct0, dct1, lt = {}, {}, []
        dct0, dct1, lt0 = ocrLabel(self.args, self.dataLabel)
        
        if self.sOrted:
            lt = sorted(lt0, reverse=True)
        else:
            lt = lt0

        nbLine = int(len(lt)/2)
        rem    = len(lt)%2
        for i in range(nbLine):
            print(dt1[3].format(lt[i][1][:40]), dt1[4].format(lt[i][0]) +'  '+dt1[3].format(lt[i+nbLine][1][:40]), dt1[4].format(lt[i+nbLine][0]))
        if rem !=0:
            print(dt1[3].format(''), dt1[4].format('') +'  '+dt1[3].format(lt[-1][1][:40]), dt1[4].format(lt[-1][0]))


    def __str__(self): # footnote [3]
        import inspect
        frame = inspect.currentframe()
        var_id = id(self.dataLabel)
        for name in frame.f_back.f_locals.keys():
            try:
                if id(eval(name)) == var_id:
                	return '< dataExplo''('' ' + name + ' '')'' >'   
            except:
                return '< dataEplo''('' ? '')'' >'




def main():
    args = parse_args()

    # load the dataset
    X_train, y_train, s_train, c_train = data_load(args, 'train.p')
    X_valid, y_valid, s_valid, c_valid = data_load(args, 'valid.p')
    X_test, y_test, s_test , c_test    = data_load(args, 'test.p')

    # [test] OK - ocrLabel(), indexClass()
    #dct0, dct1, lt = ocrLabel(args, y_valid)
    #dct0 = indexClass(y_valid)

    # [test] OK - dataExplo.showList, .showChart, .showDist
    #label = dataExplo(args, y_train)
    #label.showList(1)
    #label.showChart(2)
    #label.showDist(3)

    #print(' : {}'.format())



if __name__ == '__main__':
    main()

'''
#BACKLOG: . rewrite and clean the code
'''