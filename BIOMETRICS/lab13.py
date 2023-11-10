## Import Modules :

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
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
from sklearn.model_selection import train_test_split 



def features_vector(ft_path): 
    list_ft = []
    list_ft_dic = []
    with open(ft_path) as f: s = f.read()
    subject=0
    sessionIndex=0
    rep = 0
    KD = 0
    DDKL = 0
    UDK1K2 = 0
    HK2 = 0  
    UUKL = 0
    aux=1
    for i in range(len(s)-20):
            if (s[i+1] == 'u' or s[i+1] == 'e' or s[i+1] == 'h' or s[i+1] == 's' or s[i+1] == 'i'):
                continue
            elif(s[i] == 's'):
                subject = str(s[i+1]+s[i+2]+s[i+3])
                sessionIndex = str(s[i+5])
                if(s[i+8] == ','):
                    rep=str(s[i+7])
                    i=i+9
                elif(s[i+9] == ','):
                    rep=str(s[i+7])+str(s[i+8])
                    i=i+10
    
                if(s[i]=='-' or s[i+1]!='.'):
                    KD = str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5]+s[i+6])    
                    i=i+8
                else:
                    KD = str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5])
                    i=i+7
                if(s[i]=='-' or s[i+1]!='.'):
                    DDKL = str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5]+s[i+6])
                    i=i+8
                else:  
                    DDKL = str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5])
                    i=i+7
                if(s[i]=='-' or s[i+1]!='.'):
                    UDK1K2=float(str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5]+s[i+6]))
                    i=i+1
                else: 
                    UDK1K2=float(str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5]))
                if(s[i]=='-' or s[i+1]!='.'):
                    HK2 = float(str(s[i+7]+s[i+8]+s[i+9]+s[i+10]+s[i+11]+s[i+12]+s[i+13]))
                else:    
                    HK2 = float(str(s[i+7]+s[i+8]+s[i+9]+s[i+10]+s[i+11]+s[i+12]))    
                UUKL = str(UDK1K2+HK2)
                #print(subject+" "+sessionIndex+" "+rep+" "+KD+" "+DDKL+" "+str(float("{0:.4f}".format(float(UUKL)))))
                list_ft.append(((subject),(sessionIndex),(rep),(KD),(DDKL),(str(float("{0:.4f}".format(float(UUKL)))))))
                if(sessionIndex=='8' and rep=='50'):
                    feat.append(subject)
                    list_ft_dic.append(list_ft)
                    list_ft = []
    
    return list_ft_dic 





##########################
#Train:
    
#1-choose a specific angle of view
angle_view = 90
#persons id's:001-124
train_x=[]
train_y=[]

feat = []
file_features_path = 'DSL-StrongPasswordData.csv'
ft=features_vector(file_features_path) #maybe return x and y vectors
#print(feat)

train_KD = []
test_KD = []
KD = []
y_KD = []
train_DDKL = []
test_DDKL = []
DDKL = []
y_DDKL = []
train_UUKL = []
test_UUKL = []
UUKL = []
y_UUKL = []

gm_KD = GaussianMixture(n_components=51)
gm_DDKL = GaussianMixture(n_components=51)
gm_UUKL = GaussianMixture(n_components=51)



for i in range(len(ft)):
    for j in range(len(ft[0])):
        KD.append(ft[i][j][3])
        y_KD.append(ft[i][j][0])

    for j in range(len(ft[0])):
        DDKL.append(ft[i][j][4])
        y_DDKL.append(ft[i][j][0])

    for j in range(len(ft[0])):
        UUKL.append(ft[i][j][5])
        y_UUKL.append(ft[i][j][0])

    
train_KD,test_KD,y_train_KD,y_test_KD = train_test_split(KD,y_KD,test_size=0.2)     
gm_KD.fit(np.array(train_KD).reshape(-1, 1),y_train_KD)    

train_DDKL,test_DDKL,y_train_DDKL,y_test_DDKL = train_test_split(DDKL,y_DDKL,test_size=0.2)     
gm_DDKL.fit(np.array(train_DDKL).reshape(-1, 1),y_train_DDKL)

train_UUKL,test_UUKL,y_train_UUKL,y_test_UUKL = train_test_split(UUKL,y_UUKL,test_size=0.2)     
gm_UUKL.fit(np.array(train_UUKL).reshape(-1, 1),y_train_UUKL)    



predictions_KD = gm_KD.predict(np.array(test_KD).reshape(-1,1))
samples_number = len(predictions_KD)
sucess = 0

for t in range(len(predictions_KD)):
    #print("Predict:",feat[predictions_KD[t]]," corresponds to Train:",y_test_KD[t])
    if feat[predictions_KD[t]] == y_test_KD[t]:
        sucess = sucess+1

print("Accuracy KD = ",((samples_number-sucess)/samples_number)*100,"%") 


predictions_DDKL = gm_DDKL.predict(np.array(test_DDKL).reshape(-1,1))
samples_number = len(predictions_DDKL)
sucess = 0

for t in range(len(predictions_DDKL)):
    #print("Predict:",feat[predictions_DDKL[t]]," corresponds to Train:",y_test_DDKL[t])
    if feat[predictions_DDKL[t]] == y_test_DDKL[t]:
        sucess = sucess+1

print("Accuracy DDKL = ",((samples_number-sucess)/samples_number)*100,"%") 


predictions_UUKL = gm_UUKL.predict(np.array(test_UUKL).reshape(-1,1))
samples_number = len(predictions_UUKL)
sucess = 0

for t in range(len(predictions_UUKL)):
    #print("Predict:",feat[predictions_UUKL[t]]," corresponds to Train:",y_test_UUKL[t])
    if feat[predictions_UUKL[t]] == y_test_UUKL[t]:
        sucess = sucess+1

print("Accuracy UUKL = ", ((samples_number-sucess)/samples_number)*100,"%") 


# The most of the times UUKL is the one with best stats





# x_train_KD,x_test_KD,y_train_KD,y_test_KD = train_test_split(KD,ft_y,test_size=0.2)
# x_train_DDKL,x_test_DDKL,y_train_DDKL,y_test_DDKL = train_test_split(DDKL,ft_y,test_size=0.2)
# x_train_UUKL,x_test_UUKL,y_train_UUKL,y_test_UUKL = train_test_split(UUKL,ft_y,test_size=0.2)

# print(np.array(x_train_DDKL))
# gm_KD = GaussianMixture(n_components=3)
# gm_KD.fit(np.array(x_train_KD).reshape(-1,1),y_train_KD)

# gm_DDKL = GaussianMixture(n_components=3)
# gm_DDKL.fit(np.array(x_train_DDKL).reshape(-1,1),y_train_DDKL)

# gm_UUKL = GaussianMixture(n_components=3)
# gm_UUKL.fit(np.array(x_train_UUKL).reshape(-1,1),y_train_UUKL)



# predictions_KD = gm_KD.predict(np.array(x_test_KD).reshape(-1,1)) #np.array(x_test_KD).reshape(-1,1)
# predictions_DDKL = gm_DDKL.predict(np.array(x_test_DDKL).reshape(-1,1))
# predictions_UUKL = gm_UUKL.predict(np.array(x_test_UUKL).reshape(-1,1))


    
    
# train_x.append(g)
# train_y.append(i)
        
        
# #2-split sequences into train and val sets we can use walking status
# #train = ['bg-01', 'bg-02', 'cl-01', 'nm-03', 'nm-04', 'nm-05', 'nm-06']
# #val = ['cl-02', 'nm-01' , 'nm-02']



# clf = RandomForestClassifier(n_estimators=500, n_jobs=-1,
# random_state=2016, verbose=1, max_depth=100, max_features=100)
# clf.fit(train_x,train_y)

# val_x=[]
# val_y=[]
# for i in range(1, 5):
#     for j in val:
#         training_set_path = str(path_database+str(i).zfill(3)+'/'+j+'/'+str(angle_view).zfill(3)+'/')
#         g=GaitEnergyImage(training_set_path)
#         val_x.append(g)
#         val_y.append(i)

# predicts = clf.predict(val_x)
# #print(val_y)
# #print(predicts)
# sucess = 0
# samples_number = len(predicts)

# for t in range(len(predicts)):
#     print("Predict:",predicts[t]," corresponds to Train:",val_y[t])
#     if val_y[t] == predicts[t]:
#         sucess = sucess+1
 