## Import Modules :

import os
import cv2
import numpy as np
import math as m
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
import warnings

data_path_ridb = "RIDB"
data_path_train = "./RIDB"

filename_list_train = [f for f in os.listdir(data_path_train) if os.path.isfile(os.path.join(data_path_train, f))]


#get the skeleton of the preserved objects
def get_skeleton(img_obj_filtered):
    skel_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    skel = np.zeros(img_obj_filtered.shape, np.uint8)
    while True:
        opened = cv2.morphologyEx(img_obj_filtered, cv2.MORPH_OPEN,
        skel_element)
        temp = cv2.subtract(img_obj_filtered, opened)
        eroded = cv2.erode(img_obj_filtered, skel_element)
        skel = cv2.bitwise_or(skel, temp)
        img_obj_filtered = eroded.copy()
        if cv2.countNonZero(img_obj_filtered)==0:
            break


for filename in filename_list_train:

    I = cv2.imread(os.path.join(data_path_train, filename))
    # for f in range(len(ridb)):
    #I = Image.open(ridb[f])
    #I = cv2.imread(ridb[f])
    cv2.imshow("I",I)
    cv2.waitKey()
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    img_floodfill = gray.copy()
    h, w = img_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_floodfill,mask,(0,0), 255)
    dil_kernel = np.ones((13, 13),np.uint8)
    img_dilation = cv2.dilate(gray,dil_kernel,iterations=1)
    cv2.imshow("I1",img_dilation)
    mask_inv = cv2.bitwise_not(img_dilation)
    cv2.imshow("I2",mask_inv)
    ret,thresh = cv2.threshold(mask_inv,100,255,cv2.THRESH_BINARY)
    cv2.imshow("I Transformed",thresh)
    cv2.waitKey()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    cv2.imshow("I T1",cl1)
    n_img = 255-cl1
    cv2.imshow("I T2",n_img)
    cl2 = clahe.apply(n_img)
    cv2.imshow("I T3",cl2)
    cv2.waitKey()
    img_blur = cv2.GaussianBlur(cl2, (7,7),0)
    th3 = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,0)
    img_withmask=cv2.bitwise_and(th3, th3,mask=thresh)
    img_withmask = cv2.medianBlur(img_withmask,5)
    img_withmask = cv2.bitwise_not(img_withmask)
    closing = cv2.morphologyEx(img_withmask, cv2.MORPH_CLOSE, dil_kernel)
    img_closed_inv=255-closing
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_closed_inv, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # Filter using contour area and remove small noise
    cnts = cv2.findContours(img_closed_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:
            cv2.drawContours(img_closed_inv, [c], -1, (0,0,0), -1) 
    cv2.imshow("I Prep",img_closed_inv)
    cv2.waitKey()
    get_skeleton(img_closed_inv)
    cv2.destroyAllWindows()

