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

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os




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
            

images=[]
for i in range(len(labelsf)):
    image = cv2.imread('caltech/'+imagesf[i]) #colored image
    crop_img = image[int(imagesData['SubDir_Data'][3][i]):int(imagesData['SubDir_Data'][7][i]), int(imagesData['SubDir_Data'][2][i]):int(imagesData['SubDir_Data'][6][i])]
    resized_image = cv2.resize(crop_img, (160, 160))
    roi_float = np.array(resized_image) / 255.0
    roi_tensor = torch.from_numpy(roi_float).permute(2, 0, 1).float()
    images.append(roi_tensor)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device)) 

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('caltech/..')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=len(np.asarray(imagesf)))

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])







'''   

labellist=np.array(labellist).flatten()   

train_images = []
train_labels = []
test_images = []
test_labels = [] 

train_labels, test_labels, train_images, test_images = train_test_split(labellist, images, test_size=0.25)

FR1 = cv2.face.EigenFaceRecognizer_create()
FR2 = cv2.face.FisherFaceRecognizer_create()
FR3 = cv2.face.LBPHFaceRecognizer_create()

FR1.train(train_images, np.array(train_labels))
FR2.train(train_images, np.array(train_labels))
FR3.train(train_images, np.array(train_labels)) 
    # i=i+1
    
predict_FR1=0
predict_FR2=0
predict_FR3=0

for i in range(len(test_images)):
    img = test_images[i].copy()
    
    fr1test=int( FR1.predict(img)[0] )
    fr2test=int( FR2.predict(img)[0] )
    fr3test=int( FR3.predict(img)[0] )

    if (int(test_labels[i])-fr1test)!=0:
            predict_FR1=predict_FR1+1

    if (int(test_labels[i])-fr2test)!=0:
            predict_FR2=predict_FR2+1

    if (int(test_labels[i])-fr3test)!=0:
            predict_FR3=predict_FR3+1
    
print("The closer to 0, the better the method.")
print("Eigen Face Recognizer method:",predict_FR1,".")
print("Fisher Face Recognizer method:", predict_FR2,".")
print("LBPH Face Recognizer method:",predict_FR3,".") '''