import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2 as cv
from sklearn.metrics import classification_report



filename_list = [f for f in listdir('DB1_B') if isfile(join('DB1_B', f))]



base = {}
suspects = {}

# Initiate ORB detector
orb = cv.ORB_create()
sift = cv.SIFT_create()
feat_dict = {'sift': ' ', 'orb': ' '}
features_base=[] 
features_suspects =[]
y_pred_orb = list()
y_pred_sift = list()
for f in filename_list:
        img = cv2.imread('DB1_B/'+f)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # find and compute the keypoints with SIFT    
        kp1, des1 = sift.detectAndCompute(gray,None)

        # find and compute the keypoints with ORB
        kp2, des2 = orb.detectAndCompute(gray,None)
        feat_dict = {'sift': des1, 'orb': des2}
        if(f[4]=='1'):
            features_base.insert(int(f[0]+f[1]+f[2]), feat_dict)
            print('BASE:',int(f[0]+f[1]+f[2])) 
        else:
            features_suspects.insert(int(f[0]+f[1]+f[2]+f[4]), feat_dict)
            #print('SUS:',int(f[0]+f[1]+f[2])) 
            y_pred_orb.append(int(f[0]+f[1]+f[2]))
            y_pred_sift.append(int(f[0]+f[1]+f[2])) 



# create BFMatcher object for ORB
bf_orb = cv.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)    
# create BFMatcher object for SIFT      
# BFMatcher object for SIFT
bf_sift = cv.BFMatcher(cv2.NORM_L2, crossCheck=False)

y_test_orb = list()
y_test_sift = list()

for i in range(len(features_suspects)):
    aux_sift=0
    aux_orb=0
    good_matches_sift = []
    good_matches_orb = []
    matches_orb_len = []
    matches_sift_len = []
    for j in range(len(features_base)):
        matches_sift=bf_sift.knnMatch(features_suspects[i].get('sift'),features_base[j].get('sift'),k=2)
        matches_orb=bf_orb.knnMatch(features_suspects[i].get('orb'),features_base[j].get('orb'),k=2)
        
        for m, n in matches_sift:
            if m.distance < 0.75 * n.distance:
                good_matches_sift.append([m])               
        
        matches_sift_len.append(len(good_matches_sift))    

        for m, n in matches_orb:
            if m.distance < 0.75 * n.distance:
                good_matches_orb.append([m])      

        matches_orb_len.append(len(good_matches_orb))  
    
    y_test_sift.append(101+matches_sift_len.index(max(matches_sift_len)))
    
    y_test_orb.append(101+matches_orb_len.index(max(matches_orb_len)))
    

print('ORB:')
print(classification_report(y_test_orb, y_pred_orb))

print('SIFT:')
print(classification_report(y_test_sift, y_pred_sift))
                     