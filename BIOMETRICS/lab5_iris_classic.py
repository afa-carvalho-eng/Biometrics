## Import Modules :

import os
import cv2
import numpy as np
import math as m
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)



## Function to remove glares :

def remove_glare(image):
    H = cv2.calcHist([image], [0], None, [256], [0, 256])

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

## Function to find the exploding circle of the pupil :

def exploding_pupil(image,x_glare,y_glare,step_seed,start_radius,end_radius,step_radius,start_angle,end_angle,step_angle):

    best_x_seed = 0
    best_y_seed = 0
    best_radius = 0
    best_difference = 0

    for x_seed in [-1,0,1] :
        for y_seed in [-1,0,1] :

            x_center = x_glare + step_seed*x_seed
            y_center = y_glare + step_seed*y_seed

            previous_brightness = None

            for radius in range (start_radius,end_radius,step_radius):

                sum_brightness = 0
                nb_angle = 0

                for angle in range (start_angle,end_angle,step_angle):

                    xi = int(x_center + radius*m.cos(m.radians(angle)))
                    yi = int(y_center - radius*m.sin(m.radians(angle)))

                    if xi < 576 :
                        if yi < 576 :
                            sum_brightness = sum_brightness + image[xi,yi]
                            nb_angle = nb_angle + 1


                mean_brightness = sum_brightness/nb_angle

                if previous_brightness != None:

                    if best_difference < abs(previous_brightness - mean_brightness):
                        best_difference = abs(previous_brightness - mean_brightness)
                        best_x_seed = x_center
                        best_y_seed = y_center
                        best_radius = radius

                previous_brightness = mean_brightness

    image_circle = np.copy(image)
    cv2.circle(image_circle,(best_x_seed,best_y_seed),best_radius,(255, 0, 0))
    cv2.imshow("best circle", image_circle)
    key = cv2.waitKey()

    return(best_x_seed,best_y_seed,image_circle,best_radius)

## Function to find the exploding circle of the iris :

def exploding_iris(image,x_pupil,y_pupil,step_seed,start_radius,end_radius,step_radius,start_angle,end_angle,step_angle,bin):

    best_x_seed = 0
    best_y_seed = 0
    best_radius = 0
    best_difference = 0

    for x_seed in [-1,0,1] :
        for y_seed in [-1,0,1] :

            x_center = x_pupil + step_seed*x_seed
            y_center = y_pupil + step_seed*y_seed

            previous_brightness = None

            for radius in range (start_radius,end_radius,step_radius):

                sum_brightness = 0

                if bin == 0 :
                    # RIGHT SIDE :

                    nb_angle_right = 0

                    for angle in range (start_angle,end_angle,step_angle):


                        xi = int(x_center + radius*m.cos(m.radians(angle)))
                        yi = int(y_center - radius*m.sin(m.radians(angle)))

                        if xi < 576 :
                            if yi < 576 :
                                sum_brightness = sum_brightness + image[xi,yi]

                        nb_angle_right = nb_angle_right + 1


                    mean_brightness = sum_brightness/nb_angle_right

                    if previous_brightness != None:

                        if best_difference < abs(previous_brightness - mean_brightness):
                            best_difference = abs(previous_brightness - mean_brightness)
                            best_x_seed = x_center
                            best_y_seed = y_center
                            best_radius = radius

                    previous_brightness = mean_brightness

                if bin == 1 :
                    # LEFT SIDE :

                    nb_angle_left = 0

                    for angle in range (-end_angle,-start_angle,step_angle):

                        xi = int(x_center + radius*m.cos(m.radians(angle)))
                        yi = int(y_center - radius*m.sin(m.radians(angle)))

                        if xi < 576 :
                            if yi < 576 :
                                sum_brightness = sum_brightness + image[xi,yi]

                        nb_angle_left = nb_angle_left + 1


                    mean_brightness = sum_brightness/nb_angle_left

                    if previous_brightness != None:

                        if best_difference < abs(previous_brightness - mean_brightness):
                            best_difference = abs(previous_brightness - mean_brightness)
                            best_x_seed = x_center
                            best_y_seed = y_center
                            best_radius = radius

                    previous_brightness = mean_brightness

    image_circle = np.copy(image)
    #cv2.circle(image_circle,(best_x_seed,best_y_seed),best_radius,(255, 0, 0))
    #cv2.imshow("best circle", image_circle)
    #key = cv2.waitKey()

    return(best_x_seed,best_y_seed,image_circle,best_radius)


## Function of Gabor

def gabor_filter (image,pupil_radius,iris_radius,pupil_center,iris_center):

    kernels = []
    sigma = 2
    for theta in range(8):
        t = theta / 8. * np.pi
        kernel = (gabor_kernel(0.15, theta=t, sigma_x=sigma, sigma_y=sigma))
        kernels.append(kernel)


    step_radius_right = int((iris_radius[0]-pupil_radius)/9)+1

    step_radius_left = int((iris_radius[1]-pupil_radius)/9)+1

    step_angle = int((90)/9)+1

    X_gabor = []

    Y_gabor = []

    for radius in range (pupil_radius + step_radius_right, iris_radius[0], step_radius_right) :

        for angle in range (-45 + int(step_angle/2),45,step_angle) :

            x_gabor = pupil_center[0] + pupil_radius*m.cos(m.radians(angle)) + radius*m.cos(m.radians(angle))
            X_gabor.append(x_gabor)

            y_gabor = pupil_center[1] - pupil_radius*m.sin(m.radians(angle)) - radius*m.sin(m.radians(angle))
            Y_gabor.append(y_gabor)


    for radius in range (pupil_radius + step_radius_left, iris_radius[1], step_radius_left) :

        for angle in range (135 + int(step_angle/2),225,step_angle) :

            x_gabor = pupil_center[0] + pupil_radius*m.cos(m.radians(angle)) + radius*m.cos(m.radians(angle))
            X_gabor.append(x_gabor)

            y_gabor = pupil_center[1] - pupil_radius*m.sin(m.radians(angle)) - radius*m.sin(m.radians(angle))
            Y_gabor.append(y_gabor)

    REAL = []
    IMAG = []

    for i in range (len(X_gabor)):

        start_x = int(X_gabor[i] - 10.5)
        start_y = int(Y_gabor[i] - 10.5)
        end_x = int(X_gabor[i] + 10.5)
        end_y = int(Y_gabor[i] + 10.5)

        for kernel in kernels :

            result = (ndi.convolve(image[start_y:end_y,start_x:end_x], kernel))

            sum_result = np.sum(result)

            if sum_result.real >= 0 :
                REAL.append(1)
            else :
                REAL.append(0)

            if sum_result.imag >= 0 :
                IMAG.append(1)
            else :
                IMAG.append(0)

    return (REAL,IMAG)

## Function to know the pourcentage of similarity of 2 iris

def compare(RESULT_1,RESULT_2):

    stat = 0

    for i in range(len(RESULT_1[0])):
        if RESULT_1[0][i] == RESULT_2[0][i]:
            stat = stat + 1

        if RESULT_1[1][i] == RESULT_2[1][i]:
            stat = stat + 1

    return (stat/2048)

## Function main :

def main(t):

    # TRAIN PART :
    data_path_train = "./iris_database_train"

    filename_list_train = [f for f in os.listdir(data_path_train) if os.path.isfile(os.path.join(data_path_train, f))]

    TRAIN_DATABASE = {'name':[],'code':[]}

    for filename in filename_list_train:

        img = cv2.imread(os.path.join(data_path_train, filename))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_no_glare, x, y = remove_glare(gray)

        pupil_1 = exploding_pupil(img_no_glare,x,y,5,45,120,5,0,360,5)

        pupil_2 = exploding_pupil(img_no_glare,pupil_1[0],pupil_1[1],2,45,90,5,0,360,5)

        iris_right = exploding_iris(pupil_2[2],pupil_2[0],pupil_2[1],5,pupil_2[3]+150,300,5,-45,45,5,0)

        iris_left = exploding_iris(pupil_2[2],pupil_2[0],pupil_2[1],5,pupil_2[3]+150,300,5,135,225,5,1)

        image_two_circles = [iris_right[2],iris_left[2]]
        pupil_center = [pupil_2[0],pupil_2[1]]
        pupil_radius = pupil_2[3]
        iris_center = [iris_right[0],iris_right[1],iris_left[0],iris_left[1]]
        iris_radius = [iris_right[3],iris_left[3]]

        TRAIN_DATABASE['name'].append(filename)
        TRAIN_DATABASE['code'].append(gabor_filter(gray,pupil_radius,iris_radius,pupil_center,iris_center))

        # key = cv2.waitKey()
        # if key == ord("x"):
        #     break

    # Does it work ?

    SIMILAR_MATRICE_TRAIN = [[0 for i in range(len(filename_list_train)) ] for i in range(len(filename_list_train))]

    for i in range(len(filename_list_train)) :
        for j in range(len(filename_list_train)) :
            SIMILAR_MATRICE_TRAIN[i][j] = compare(TRAIN_DATABASE['code'][i],TRAIN_DATABASE['code'][j])

    # TEST PART :
    data_path_test = "C:/Users/afaca/Desktop/BIOMETRICS/iris_database_test"

    filename_list_test = [f for f in os.listdir(data_path_test) if os.path.isfile(os.path.join(data_path_test, f))]

    TEST_DATABASE = {'name':[],'code':[]}

    for filename in filename_list_test:

        img = cv2.imread(os.path.join(data_path_test, filename))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_no_glare, x, y = remove_glare(gray)

        pupil_1 = exploding_pupil(img_no_glare,x,y,5,45,120,5,0,360,5)

        pupil_2 = exploding_pupil(img_no_glare,pupil_1[0],pupil_1[1],2,45,90,5,0,360,5)

        iris_right = exploding_iris(pupil_2[2],pupil_2[0],pupil_2[1],5,pupil_2[3]+150,300,5,-45,45,5,0)

        iris_left = exploding_iris(pupil_2[2],pupil_2[0],pupil_2[1],5,pupil_2[3]+150,300,5,135,225,5,1)

        image_two_circles = [iris_right[2],iris_left[2]]
        pupil_center = [pupil_2[0],pupil_2[1]]
        pupil_radius = pupil_2[3]
        iris_center = [iris_right[0],iris_right[1],iris_left[0],iris_left[1]]
        iris_radius = [iris_right[3],iris_left[3]]

        TEST_DATABASE['name'].append(filename)
        TEST_DATABASE['code'].append(gabor_filter(gray,pupil_radius,iris_radius,pupil_center,iris_center))

        # key = cv2.waitKey()
        # if key == ord("x"):
        #     break

    # RECOGNITION :

    SIMILAR_MATRICE_TEST = [[0 for i in range(len(filename_list_train)) ] for i in range(len(filename_list_test))]

    for i in range(len(filename_list_test)) :
        for j in range(len(filename_list_train)) :
            SIMILAR_MATRICE_TEST[i][j] = compare(TEST_DATABASE['code'][i],TRAIN_DATABASE['code'][j])

        result = np.max(SIMILAR_MATRICE_TEST[i])
        index = SIMILAR_MATRICE_TEST[i].index(result)

        if result > t :
            print("L'image" + TEST_DATABASE['name'][i] + "ressemble à l'imgage" + TRAIN_DATABASE['name'][index] )
        else :
            print("L'image" + TEST_DATABASE['name'][i] + "ne fait pas parti de la base de donnée" )




if __name__ == "__main__":
    main(0.9)


