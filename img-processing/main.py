import cv2
import glob
import numpy as np

PATH = 'C:/Users/LENOVO/Documents/Informatics/Semester 5/Kecerdasan Buatan/Tubes/Project AI/Dataset/'


def belimbing_adjustment():
    path = PATH + 'Belimbing/'
    format = ['png', 'jpg']
    files = []
    [files.extend(glob.glob(path + '*.' + e)) for e in format]
    images = [cv2.imread(file) for file in files]

    i = 1
    for img in images:
        im_adjusted = cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, 0)
        im_name = PATH + 'Belimbing_contrast/' + str(i) + '.jpg'
        cv2.imwrite(im_name, im_adjusted)
        i += 1


def seledri_adjustment():
    path = PATH + 'Seledri/'
    format = ['png', 'jpg']
    files = []
    [files.extend(glob.glob(path + '*.' + e)) for e in format]
    images = [cv2.imread(file) for file in files]

    i = 1
    for img in images:
        im_adjusted = cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, 0)
        im_name = PATH + 'Seledri_contrast/' + str(i) + '.jpg'
        cv2.imwrite(im_name, im_adjusted)
        i += 1


belimbing_adjustment()
seledri_adjustment()
