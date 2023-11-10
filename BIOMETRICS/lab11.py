## Import Modules :

import os
from os import listdir
from os.path import isfile, join
from cv2 import IMREAD_GRAYSCALE
import librosa
import cv2
import numpy as np
import math as m
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier 



def GaitEnergyImage(training_set_path): 
    list_g = []
    training_set = [f for f in listdir(training_set_path) if isfile(join(training_set_path, f))]

    for f in training_set:
        
        im = cv2.imread(training_set_path+f,cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #print("Number of contours:" + str(len(contours)))
        if len(contours)!=0: 
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            im_2=im.copy()
            cv2.rectangle(im_2,(x,y),(x+w,y+h),(255,0,0),1)
            cv2.imshow("result",im_2)
            cv2.waitKey(50)
            cv2.destroyAllWindows()
            new_im = im[y:y+h,x:x+w]
            cv2.imshow("result",new_im)
            cv2.waitKey(50)
            cv2.destroyAllWindows()
            resized=cv2.resize(new_im,(70,210))
            resized = resized.astype('float64')
            list_g.append(resized)
    gei = np.zeros((70,210))
    gei = np.mean(list_g, axis=0)
    return gei.flatten()




path_database='GaitDatasetB-silh/'
train = ['nm-03', 'bg-01', 'nm-06', 'nm-04', 'cl-01']

val = ['nm-02', 'nm-05', 'bg-02']

##########################
#Train:
    
#1-choose a specific angle of view
angle_view = 90
#persons id's:001-124
train_x=[]
train_y=[]
for i in range(1, 5): #person 1 to 5
    for j in train:
        training_set_path = str(path_database+str(i).zfill(3)+'/'+j+'/'+str(angle_view).zfill(3)+'/')
        g=GaitEnergyImage(training_set_path)
        train_x.append(g)
        train_y.append(i)
        
        
#2-split sequences into train and val sets we can use walking status
#train = ['bg-01', 'bg-02', 'cl-01', 'nm-03', 'nm-04', 'nm-05', 'nm-06']
#val = ['cl-02', 'nm-01' , 'nm-02']



clf = RandomForestClassifier(n_estimators=500, n_jobs=-1,
random_state=2016, verbose=1, max_depth=100, max_features=100)
clf.fit(train_x,train_y)

val_x=[]
val_y=[]
for i in range(1, 5):
    for j in val:
        training_set_path = str(path_database+str(i).zfill(3)+'/'+j+'/'+str(angle_view).zfill(3)+'/')
        g=GaitEnergyImage(training_set_path)
        val_x.append(g)
        val_y.append(i)

predicts = clf.predict(val_x)
print(val_y)
print(predicts)
sucess = 0
samples_number = len(predicts)

for t in range(len(predicts)):
    print("Predict:",predicts[t]," corresponds to Train:",val_y[t])
    if val_y[t] == predicts[t]:
        sucess = sucess+1

print("Accuracy = ",(sucess/samples_number)*100,"%")