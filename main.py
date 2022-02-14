#!/usr/bin/env python

import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import os
import xml.etree.ElementTree as ET

#def load_data(path, failname):
#    data = []
#    return data

#def main():
#    return

#if __name__ == '__main__':
#    main()

import sys
import cv2 as cv
import numpy as np

def loadData(path,filename):
    data = []
    for file in os.listdir(os.path.join(path, filename)):
        f = os.path.join(os.path.join(path, filename),file)
        if f.endswith(".xml"):
            XMLtree = ET.parse(f)
            XMLroot = XMLtree.getroot()
            imgName = XMLroot[1].text
            imgClassID = XMLroot[4][0].text
            if os.path.isfile(os.path.join(os.path.join(path, 'images'),imgName)):
               data.append(ImageAnalizer(path, imgName, imgClassID))
    return data


class ImageAnalizer:
    def __init__(self, path, filename, classID):
        self.classID = classID
        self.name = filename
        self.img = cv2.imread(os.path.join(os.path.join(path, 'images'),filename))

def main(argv):
    dataTrain = loadData('./train/', 'annotations')
    dataTest = loadData('./test/', 'annotations')
    iter = 0
    for i in dataTrain:
        iter = iter + 1
    print(iter)
    '''
    default_file = 'img.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    #gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=10, param2=10,
                              minRadius=1, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)

    cv.imshow("detected circles", src)
    cv.waitKey(0)
    '''
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])