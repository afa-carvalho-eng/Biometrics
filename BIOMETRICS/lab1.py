import csv
from cv2 import imread
import numpy as np
import scipy.io
import glob
import os
import numpy
from sklearn import linear_model
import cv2
from cv2 import imshow, imread, resize

from fileinput import filename
# from genericpath import isfile
# from ntpath import join
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split





filename_list = [f for f in listdir('./caltech') if isfile(join('./caltech', f))]
lastrow=0
aux=0
count=0
imagesf = []

with open('caltech/caltech_labels.csv', newline='') as csvfile:
        labels = csv.reader(csvfile, delimiter=' ')
        labelsf = []
        labellist = []
        for row in labels:
            count=count+1
            if lastrow == row:
                aux=aux+1
            elif lastrow != row:
                if aux >= 19:
                    if lastrow != 1:
                        count=count-aux
                    while aux>0 :
                        labellist.append([int(i) for i in lastrow])
                        labelsf.append(count)
                        aux=aux-1
                        count=count+1
                lastrow = row
                aux = 1


imagesData = scipy.io.loadmat('caltech/ImageData.mat')

resized_images = []

for f in filename_list:
    if f[0] == 'i':
        number = int(f[7]+f[8]+f[9])
        if number in labelsf:
            imagesf.append(f)
            
         
for i in range(len(labelsf)):
    image = cv2.imread('caltech/'+imagesf[i],0) #gray image
    crop_img = image[int(imagesData['SubDir_Data'][3][i]):int(imagesData['SubDir_Data'][7][i]), int(imagesData['SubDir_Data'][2][i]):int(imagesData['SubDir_Data'][6][i])]
    resized_images.append(cv2.resize(crop_img, (70, 100)))
    
labellist=np.array(labellist).flatten()   

train_images = []
train_labels = []
test_images = []
test_labels = [] 

train_labels, test_labels, train_images, test_images = train_test_split(labellist, resized_images, test_size=0.25)

FR1 = cv2.face.EigenFaceRecognizer_create()
FR2 = cv2.face.FisherFaceRecognizer_create()
FR3 = cv2.face.LBPHFaceRecognizer_create()

FR1.train(train_images, np.array(train_labels))
FR2.train(train_images, np.array(train_labels))
FR3.train(train_images, np.array(train_labels)) 

    
predict_FR1=0
predict_FR2=0
predict_FR3=0

for i in range(len(test_images)):
    img = test_images[i].copy()
    
    fr1test=int( FR1.predict(img)[0] )
    fr2test=int( FR2.predict(img)[0] )
    fr3test=int( FR3.predict(img)[0] )

    if (int(test_labels[i])!=fr1test):
            predict_FR1=predict_FR1+1

    if (int(test_labels[i])!=fr2test):
            predict_FR2=predict_FR2+1

    if (int(test_labels[i])!=fr3test):
            predict_FR3=predict_FR3+1
    
    predict1=(predict_FR1/len(test_images))*100
    predict2=(predict_FR2/len(test_images))*100
    predict3=(predict_FR3/len(test_images))*100

print("Accuracy of used methods:")
print("The accuracy of Eigen Face Recognizer method is: {:.1f}".format(predict1),"%")
print("The accuracy of Fisher Face Recognizer method is: {:.1f}".format(predict2),"%")
print("The accuracy of LBPH Face Recognizer method is: {:.1f}".format(predict3),"%")