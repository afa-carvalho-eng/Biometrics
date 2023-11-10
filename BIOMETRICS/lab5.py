from email.mime import image
import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import math 
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

img_no_glare_l=[]


def remove_glare(image):
    H = cv2.calcHist([image], [0], None, [256], [0, 256])
    # plt.plot(H[150:])
    # plt.show()
    idx = np.argmax(H[150:]) + 151
    binary = cv2.threshold(image, idx, 255, cv2.THRESH_BINARY)[1]

    st3 = np.ones((3, 3), dtype="uint8")
    st7 = np.ones((7, 7), dtype="uint8")

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, st3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, st3, iterations=2)

    im_floodfill = binary.copy()

    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = binary | im_floodfill_inv
    im_out = cv2.morphologyEx(im_out, cv2.MORPH_DILATE, st7, iterations=1)
    _, _, stats, cents = cv2.connectedComponentsWithStats(im_out)
    cx, cy = 0, 0
    for st, cent in zip(stats, cents):
        if 1500 < st[4] < 3000:
            if 0.9 < st[2] / st[3] < 1.1:
                cx, cy = cent.astype(int)
                r = st[2] // 2
                cv2.circle(image, (cx, cy), r, (125, 125, 125), thickness=2)

    image = np.where(im_out, 80, image)
    image = cv2.medianBlur(image, 5)

    return image, cx, cy

def exploding_pupil(img,x,y):
    
    best_x = 0
    best_y = 0
    best_radius = 0
    best_diff = 0 
    for x_seed in range(-1,2):
        for y_seed in range(-1,2):
            x_center = x + 5*x_seed
            y_center = y + 5*y_seed
            brightness = 0
            circles = []
            for radius in range(50,200,5):
                brightness_tot = 0
                angle_tot = 0
                for angle in range(0,360,5):
                    dist_x = int(x + radius*math.cos(angle))
                    dist_y = int(y + radius*math.sin(angle))
                    if dist_x < 576: #axis 0 with size 576
                        if dist_y< 576:
                            angle_tot = angle_tot+1
                            brightness_tot = brightness_tot+img[dist_x,dist_y]
                brightness_mean = brightness_tot/angle_tot

                if brightness != 0:
                    abs_diff = abs(brightness-brightness_mean)
                    if best_diff < abs_diff:
                        best_diff = abs_diff
                        best_x = x
                        best_y = y
                        best_radius = radius

                brightness = brightness_mean                                    
    
    #print(best_y, best_x)
    img_circle = cv2.circle(img,(best_x,best_y),best_radius,(255, 255, 0),2)
    cv2.imshow("Final circle Pupil", img_circle)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return(best_x,best_y,img_circle,best_radius)

def exploding_iris(img,x,y, start_radius,side):
    
    best_x = 0
    best_y = 0
    best_radius = 0
    best_diff = 0 
    for x_seed in range(-1,2):
        for y_seed in range(-1,2):
            x_center = x + 5*x_seed
            y_center = y + 5*y_seed
            brightness = 0
            circles = []
            for radius in range(start_radius,200,5):
                brightness_tot = 0
                if side == 1: #Right side:

                    angle_tot = 0
                    for angle in range(0,360,5):
                        dist_x = int(x + radius*math.cos(angle))
                        dist_y = int(y + radius*math.sin(angle))
                        if dist_x < 576: #axis 0 with size 576
                            if dist_y< 576:
                                angle_tot = angle_tot+1
                                brightness_tot = brightness_tot + img[dist_x,dist_y]
                    brightness_mean = brightness_tot/angle_tot

                    if brightness != 0:
                        abs_diff = abs(brightness-brightness_mean)
                        if best_diff < abs_diff:
                            best_diff = abs_diff
                            best_x = x
                            best_y = y
                            best_radius = radius

                    brightness = brightness_mean 

            if side == 2: #Left side:

                    angle_tot = 0
                    for angle in range(0,360,5):
                        dist_x = int(x + radius*math.cos(angle))
                        dist_y = int(y + radius*math.sin(angle))
                        if dist_x < 576: #axis 0 with size 576
                            if dist_y< 576:
                                angle_tot = angle_tot+1
                                brightness_tot = brightness_tot+img[dist_x,dist_y]
                    brightness_mean = brightness_tot/angle_tot

                    if brightness != 0:
                        abs_diff = abs(brightness-brightness_mean)
                        if best_diff < abs_diff:
                            best_diff = abs_diff
                            best_x = x
                            best_y = y
                            best_radius = radius

                    brightness = brightness_mean 

    print(best_y, best_x)
    img_circle = cv2.circle(img,(best_x,best_y),best_radius,(255, 255, 0),2)
    cv2.imshow("Final circle Iris", img_circle)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return(best_x,best_y,img_circle,best_radius)

def main(data_path_train, data_path_test):
    # Get files from data path
    filename_list_train = [
        f for f in os.listdir(data_path_train) if os.path.isfile(os.path.join(data_path_train, f))
    ]
    filename_list_test = [
        f for f in os.listdir(data_path_test) if os.path.isfile(os.path.join(data_path_test, f))
    ]
    train_list=[]
    train_list_names=[]
    test_list=[]
    test_list_names = []

    for filename in filename_list_train:
        # Read image
        img = cv2.imread(os.path.join(data_path_train, filename))

        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove glare
        img_no_glare, x, y = remove_glare(gray)
        #img_no_glare_l.append(img_no_glare)
        # Exploding circle algorithm
        # TODO

        pupil_1 = exploding_pupil(img_no_glare,x,y)
        pupil_2 = exploding_pupil(img_no_glare,pupil_1[0],pupil_1[1])
        pupil_center =[pupil_2[0],pupil_2[1]]
        pupil_radius =pupil_2[3]

        iris_r = exploding_iris(pupil_2[2],pupil_2[0],pupil_2[1],pupil_2[3],1)
        iris_l = exploding_iris(pupil_2[2],pupil_2[0],pupil_2[1],pupil_2[3],2)
        iris_center = [iris_r[0],iris_r[1],iris_l[0],iris_l[0],iris_l[1]]
        iris_r = [iris_r[3],iris_l[3]]

        # Gabor filters
        # TODO
        train_list_names.append(filename)
        train_list.append(gabor_filters(gray,pupil_radius,iris_r,pupil_center,iris_center))
        cv2.imshow("Original image"+filename[4]+filename[6], img)
        cv2.imshow("Gray", gray)
        cv2.imshow("No glare", img_no_glare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    train_results = [[0 for aux in range(len(filename_list_train)) ] for aux in range(len(filename_list_train))]    
    
    for x in range(len(filename_list_train)) :
        for y in range(len(filename_list_train)) :
            train_results[x][y] = compare(train_list[x],train_list[y])

    ###################################################
    #test
     
    for filename in filename_list_test:
        # Read image
        img = cv2.imread(os.path.join(data_path_test, filename))

        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove glare
        img_no_glare, x, y = remove_glare(gray)
        #img_no_glare_l.append(img_no_glare)
        # Exploding circle algorithm
        # TODO

        pupil_1 = exploding_pupil(img_no_glare,x,y)
        pupil_2 = exploding_pupil(img_no_glare,pupil_1[0],pupil_1[1])
        pupil_center =[pupil_2[0],pupil_2[1]]
        pupil_radius =pupil_2[3]

        iris_r = exploding_iris(pupil_2[2],pupil_2[0],pupil_2[1],pupil_2[3],1)
        iris_l = exploding_iris(pupil_2[2],pupil_2[0],pupil_2[1],pupil_2[3],2)
        iris_center = [iris_r[0],iris_r[1],iris_l[0],iris_l[0],iris_l[1]]
        iris_r = [iris_r[3],iris_l[3]]

        # Gabor filters
        # TODO
        test_list_names.append(filename)
        test_list.append(gabor_filters(gray,pupil_radius,iris_r,pupil_center,iris_center))
        cv2.imshow("Original image"+filename[4]+filename[6], img)
        cv2.imshow("Gray", gray)
        cv2.imshow("No glare", img_no_glare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    test_results = [[0 for aux in range(len(filename_list_test)) ] for aux in range(len(filename_list_test))]    
    
    for x in range(len(filename_list_test)) :
        for y in range(len(filename_list_train)) :
            test_results[x][y] = compare(test_list[x],train_list[y])
     
        print(test_results)
        best_similarity = np.max(test_results[x])
        index_best = test_results[x].index(best_similarity)

        if best_similarity > 0.9 :
            print("Test: "+test_list_names[x]+" - Train: "+train_list_names[index_best])        
            print("Test image: "+test_list_names[x]+" is not part of the data base.")
    
    



    # pupil_detection(img_no_glare_l)


# def pupil_detection(img_no_glare_l)
#     for 

## Function of Gabor

def gabor_filters(img,pupil_radius,iris_radius,pupil_center, sigma):

    kernels_list = []
    for aux in range(8):
        t = float(aux / 8. * np.pi)
        gabor_k = (gabor_kernel(0.15, theta=t, sigma_x=sigma, sigma_y=sigma))
        kernels_list.append(gabor_k)


    step_radius_r = int((iris_radius[0]-pupil_radius)/9)+1
    step_radius_l = int((iris_radius[1]-pupil_radius)/9)+1

    gabor_x = []
    gabor_y = []

    for radius in range (pupil_radius + step_radius_r, iris_radius[0], step_radius_r) :
        for angle in range (-45 + int(11/2),45,11) :
            gabor_x.append(pupil_center[0] + pupil_radius*math.cos(math.radians(angle)) + radius*math.cos(math.radians(angle)))
            gabor_y.append(pupil_center[1] - pupil_radius*math.sin(math.radians(angle)) - radius*math.sin(math.radians(angle)))


    for radius in range (pupil_radius + step_radius_l, iris_radius[1], step_radius_l) :
        for angle in range (135 + int(11/2),225,11) : 
            gabor_x.append(pupil_center[0] + pupil_radius*math.cos(math.radians(angle)) + radius*math.cos(math.radians(angle))) 
            gabor_y.append(pupil_center[1] - pupil_radius*math.sin(math.radians(angle)) - radius*math.sin(math.radians(angle)))

    real = []
    imag = []

    for i in range (len(gabor_x)):
        start_x = int(gabor_x[i] - 10.5)
        start_y = int(gabor_y[i] - 10.5)
        end_x = int(gabor_x[i] + 10.5)
        end_y = int(gabor_y[i] + 10.5)

        for kernel in kernels_list :
            result = (ndi.convolve(image[start_y:end_y,start_x:end_x], kernel))
            sum_result = np.sum(result)

            if sum_result.real >= 0 :
                real.append(1)
            else :
                real.append(0)

            if sum_result.imag >= 0 :
                imag.append(1)
            else :
                imag.append(0)

    return (real,imag)


#1st attempt
# Function to know the pourcentage of similarity of 2 iris
def gabor(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def compare(term1,term2):
    result = 0

    for i in range(len(term1[0])):
        if term1[0][i] == term2[0][i]:
            result = result + 1

        if term1[1][i] == term2[1][i]:
            result = result + 1
    result=result/2048
    return (result)


if __name__ == "__main__":
    data_path_train = "./iris_database_train"
    data_path_test = "./iris_database_test"
    main(data_path_train,data_path_test)
    cv2.destroyAllWindows()
        
        