# def image_processing_function(img):
#     edges = cv.Canny(img,50,100)
#     return cv.merge((edges,edges,edges))
################ some imports ###############
import tensorflow as tf
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import pandas as pd
import cv2 as cv 
import cv2
import sys
import matplotlib.pyplot as plt
import skimage
import os
import re
from tqdm import tqdm
import numpy as np
from numpy.random import seed 
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from keras import backend as k 
from keras import optimizers
from keras import applications
from keras import layers
from keras import models
from keras import callbacks
from keras import optimizers
from keras.backend import tensorflow_backend
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications import xception
from keras.applications import resnet50
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils import np_utils
from keras.regularizers import l2
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model 
from keras.layers import Activation
from keras.layers import merge, Input
from keras.layers import LeakyReLU
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Conv2D,Dropout,BatchNormalization,Dense,MaxPooling2D,ZeroPadding2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
img_width, img_height = 224, 224
from keras.applications.imagenet_utils import preprocess_input
import keras
from keras.models import Model
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers , optimizers
from keras.layers import Input
from keras.applications import models
from keras.applications import VGG16
from keras.utils import np_utils
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model 

def f1ScoreAndConfusionMatrix(yActual,yPrediction,color='Blues'):
    """ first --> yActual
    second ---> yPrediction
    it will produce both f1 score and confusion matrix 
    return f1,cm"""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    f1 = f1_score(yActual,yPrediction)
    cm = confusion_matrix(yActual,yPrediction)
    print('\nConfusion Matrix:\n',cm)
    t1=cm[0,:]/sum(cm[0,:])
    t2=cm[1,:]/sum(cm[1,:])
    cm=np.array(list([t1,t2]))
    ax= plt.subplot()
    print('\nConfusion Matrix (as Percentage)')
    sns.heatmap(cm.astype(float), annot=True, ax = ax,cmap=color,fmt='g');
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['bad', 'good']); ax.yaxis.set_ticklabels(['bad', 'good'])
    plt.show()
    
    print('Bad data detection probability: ',cm[0][0])
    print('Good data detection probability:',cm[1][1])
    print('F1 score',f1)
    return f1,cm
def OneHotEncodeDecoder(X):
    """ imodel.add(Dense, activation='relu'))
f we give an one hot encoded version of list, it will give us the original list"""
    import numpy as np
    return [np.argmax(i) for i in X]
def make_directory(directory_name):
    try:
        os.makedirs(directory_name)
    except:FileExistsError
def remove_directory_contents(dir_name):
    from shutil import rmtree
    try:
        rmtree(dir_name)
        make_directory(dir_name)
    except: FileNotFoundError
def c(img):
    plt.figure(figsize=(8,8))
    plt.grid(True)
    plt.imshow(img,cmap='gray')
    plt.show()

def train_test_val_generator(num):
    """num=which open data directory"""
    from shutil import copyfile,move
    import Augmentor
    try:
        remove_directory_contents('problem'+str(num))
    except: FileNotFoundError
    try:
        os.makedirs('problem'+str(num)+'/train/good')
        os.makedirs('problem'+str(num)+'/train/bad')
        os.makedirs('problem'+str(num)+'/test/good')
        os.makedirs('problem'+str(num)+'/test/bad')
        os.makedirs('problem'+str(num)+'/val/good')
        os.makedirs('problem'+str(num)+'/val/bad')
    except: FileExistsError

    goodArr=os.listdir('Class'+str(num))
    goodArr=[i for i in goodArr if i.endswith('png')] #a condition to ignore ipython checkpoint
    goodArr=[os.path.join('Class'+str(num),i) for i in goodArr]  #to create a path
    goodArr=random.sample(goodArr,999) 
    for i in range (334):
        src=goodArr[i]
        dst=os.path.join('problem'+str(num)+'/train/good', goodArr[i][7:])
        copyfile(src,dst)

    for i in range (334,667):
        src=goodArr[i]
        dst=os.path.join('problem'+str(num)+'/test/good', goodArr[i][7:])
        copyfile(src,dst)
    for i in range (667,999):
        src=goodArr[i]
        dst=os.path.join('problem'+str(num)+'/val/good', goodArr[i][7:])
        copyfile(src,dst)

    badArr=os.listdir('Class'+str(num)+'_def')
    badArr=[i for i in badArr if i.endswith('png')] #a condition to ignore ipython checkpoint
    badArr=[os.path.join('Class'+str(num)+'_def',i) for i in badArr]  #to create a path
    random.shuffle(badArr)

    for i in range (50):
        src=badArr[i]
        dst=os.path.join('problem'+str(num)+'/train/bad', badArr[i][11:])
        copyfile(src,dst)

    for i in range (50,100):
        src=badArr[i]
        dst=os.path.join('problem'+str(num)+'/test/bad', badArr[i][11:])
        copyfile(src,dst)

    for i in range (100,150):
        src=badArr[i]
        dst=os.path.join('problem'+str(num)+'/val/bad', badArr[i][11:])
        copyfile(src,dst)


    p=Augmentor.Pipeline('problem'+str(num)+'/train/bad/')

    p.rotate90(probability=1.0)
    p.process()
    p.rotate180(probability=1.0)
    p.process()
    p.rotate270(probability=1.0)
    p.process()

    source=(os.listdir('problem'+str(num)+'/train/bad/output'))
    source=[i for i in source if i.endswith('png')]

    for i in range(len(source)):
        src=os.path.join('problem'+str(num)+'/train/bad/output',source[i])
        dst=os.path.join('problem'+str(num)+'/train/bad',src[26:])
        move(src,dst)
    os.removedirs('problem'+str(num)+'/train/bad/output')


    p=Augmentor.Pipeline('problem'+str(num)+'/test/bad/')

    p.rotate90(probability=1.0)
    p.process()
    p.rotate180(probability=1.0)
    p.process()
    p.rotate270(probability=1.0)
    p.process()


    source=(os.listdir('problem'+str(num)+'/test/bad/output'))
    source=[i for i in source if i.endswith('png')]

    for i in range(len(source)):
        src=os.path.join('problem'+str(num)+'/test/bad/output',source[i])
        dst=os.path.join('problem'+str(num)+'/test/bad',src[25:])
        move(src,dst)

    os.removedirs('problem'+str(num)+'/test/bad/output')
    p=Augmentor.Pipeline('problem'+str(num)+'/val/bad/')
    p.rotate90(probability=1.0)
    p.process()
    p.rotate180(probability=1.0)
    p.process()
    p.rotate270(probability=1.0)
    p.process()

    source=(os.listdir('problem'+str(num)+'/val/bad/output'))
    source=[i for i in source if i.endswith('png')]

    for i in range(len(source)):
        src=os.path.join('problem'+str(num)+'/val/bad/output',source[i])
        dst=os.path.join('problem'+str(num)+'/val/bad',src[24:])
        move(src,dst)
    os.removedirs('problem'+str(num)+'/val/bad/output')

    print("train_good image number: ",len(os.listdir('problem'+str(num)+'/train/good')))
    print("train_bad image number: ",len(os.listdir('problem'+str(num)+'/train/bad')))
    print("test_good image number: ",len(os.listdir('problem'+str(num)+'/test/good')))
    print("train_bad image number: ",len(os.listdir('problem'+str(num)+'/test/bad')))
    print("val_good image number: ",len(os.listdir('problem'+str(num)+'/val/good')))
    print("val_bad image number: ",len(os.listdir('problem'+str(num)+'/val/bad')))

def xTrain_xVal_xTest_from_train_test_val_folders(num,size,image_processing_function,image_processing=False):
    trainGoodArr=os.listdir('problem'+str(num)+'/train/good/')
    trainGoodArr=[i for i in trainGoodArr if i.endswith('png')]
    trainGoodArr=['problem'+str(num)+'/train/good/'+i for i in trainGoodArr]
    df1=pd.DataFrame({'paths':trainGoodArr,'result':np.ones(len(trainGoodArr))})


    trainBadArr=os.listdir('problem'+str(num)+'/train/bad/')
    trainBadArr=[i for i in trainBadArr if i.endswith('png')]
    trainBadArr=['problem'+str(num)+'/train/bad/'+i for i in trainBadArr]
    df2=pd.DataFrame({'paths':trainBadArr,'result':np.zeros(len(trainBadArr))})

    valGoodArr=os.listdir('problem'+str(num)+'/val/good/')
    valGoodArr=[i for i in valGoodArr if i.endswith('png')]
    valGoodArr=['problem'+str(num)+'/val/good/'+i for i in valGoodArr]
    df3=pd.DataFrame({'paths':valGoodArr,'result':np.ones(len(valGoodArr))})


    valBadArr=os.listdir('problem'+str(num)+'/val/bad/')
    valBadArr=[i for i in valBadArr if i.endswith('png')]
    valBadArr=['problem'+str(num)+'/val/bad/'+i for i in valBadArr]
    df4=pd.DataFrame({'paths':valBadArr,'result':np.zeros(len(valBadArr))})

    testGoodArr=os.listdir('problem'+str(num)+'/test/good/')
    testGoodArr=[i for i in testGoodArr if i.endswith('png')]
    testGoodArr=['problem'+str(num)+'/test/good/'+i for i in testGoodArr]
    df5=pd.DataFrame({'paths':testGoodArr,'result':np.ones(len(testGoodArr))})


    testBadArr=os.listdir('problem'+str(num)+'/test/bad/')
    testBadArr=[i for i in testBadArr if i.endswith('png')]
    testBadArr=['problem'+str(num)+'/test/bad/'+i for i in testBadArr]
    df6=pd.DataFrame({'paths':testBadArr,'result':np.zeros(len(testBadArr))})
    
    traindf = pd.concat([df1, df2], ignore_index=True)
    traindf = traindf.sample(len(traindf))
    valdf = pd.concat([df3, df4], ignore_index=True)
    valdf = valdf.sample(len(valdf))
    testdf = pd.concat([df5, df6], ignore_index=True)
    testdf = testdf.sample(len(testdf))

    if image_processing == False:
        xTrain = np.array([ cv.resize((cv.imread(traindf.paths.values[i])),(size,size))  for i in range(len(traindf))])
        yTrain = traindf.result.values
        yTrain=np_utils.to_categorical(yTrain, 2)

        xVal = np.array([ cv.resize((cv.imread(valdf.paths.values[i])),(size,size))  for i in range(len(valdf))])
        yVal = valdf.result.values
        yVal=np_utils.to_categorical(yVal, 2)

        xTest = np.array([ cv.resize((cv.imread(testdf.paths.values[i])),(size,size))  for i in range(len(testdf))])
        yTest = testdf.result.values
        yTest=np_utils.to_categorical(yTest, 2)
    else:
        xTrain = np.array([ cv.resize(image_processing_function(cv.imread(traindf.paths.values[i])),(size,size))  for i in range(len(traindf))])
        yTrain = traindf.result.values
        yTrain=np_utils.to_categorical(yTrain, 2)

        xVal = np.array([ cv.resize(image_processing_function(cv.imread(valdf.paths.values[i])),(size,size))  for i in range(len(valdf))])
        yVal = valdf.result.values
        yVal=np_utils.to_categorical(yVal, 2)

        xTest = np.array([ cv.resize(image_processing_function(cv.imread(testdf.paths.values[i])),(size,size))  for i in range(len(testdf))])
        yTest = testdf.result.values
        yTest=np_utils.to_categorical(yTest, 2)
        

    xTrain = xTrain/255
    xVal = xVal/255
    xTest = xTest/255
    return xTrain,yTrain,xVal,yVal,xTest,yTest