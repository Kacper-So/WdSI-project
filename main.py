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
            if os.path.isfile(os.path.join(os.path.join(path, 'images'),imgName)):
               data.append(ImageAnalizer(path,f))
    return data


class ImageAnalizer:
    def __init__(self, path, f):
        self.path = path
        self.XMLf = f

    def readXML(self):
        self.XMLtree = ET.parse(self.XMLf)
        XMLroot = self.XMLtree.getroot()
        self.classID = XMLroot[4][0].text
        self.name = XMLroot[1].text
        self.width = int(XMLroot[2][0].text)
        self.height = int(XMLroot[2][1].text)
        self.img = cv2.imread(os.path.join(os.path.join(self.path, 'images'),self.name), cv.IMREAD_COLOR)

    def findCircles(self):
        self.readXML()
        cv.imshow("detected circles", self.img)
        cv.waitKey(0)
        for iterX in range(self.width):
            for iterY in range(self.height):
                (b, g, r) = self.img[iterY, iterX]
                rFunction = max(0, min(r - g, r - b) / (r + g + b))
                if r >= g and r >= b and g/(r - g) <= 6:
                    r = rFunction * 255
                else :
                    r = 0
                self.img[iterY, iterX] = (r, r, r)
        cv.imshow("detected circles", self.img)
        cv.waitKey(0)

        segmentsDensity = 1
        for iter in range(1):
            dx = int(self.width / segmentsDensity)
            dy = int(self.height / segmentsDensity)
            for iterY in range(segmentsDensity):
                for iterX in range(segmentsDensity):
                    segment = self.img[iterY*dy:iterY*dy+dy, iterX*dx:iterX*dx+dx]
                    rows = segment.shape[0]
                    graySegment = cv.cvtColor(segment, cv.COLOR_BGR2GRAY)
                    '''
                    circles = cv.HoughCircles(graySegment, cv.HOUGH_GRADIENT, 1, rows/2, param1 = 100.0, param2 = 10.0,minRadius = int(dx/10), maxRadius = dx)
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for i in circles[0, :]:
                            center = (i[0], i[1])
                            # circle center
                            cv.circle(segment, center, 1, (0, 100, 100), 3)
                            # circle outline
                            radius = i[2]
                            cv.circle(segment, center, radius, (255, 0, 255), 3)
                    circles = None
                    '''
                    vis = self.img.copy()
                    mser = cv2.MSER_create()
                    regions, box = mser.detectRegions(graySegment)
                    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
                    cv2.polylines(vis, hulls, 1, (0, 255, 0))

                    mask = np.zeros((self.img.shape[0], self.img.shape[1], 1), dtype=np.uint8)

                    for contour in hulls:
                        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
                    text_only = cv2.bitwise_and(self.img, self.img, mask=mask)

                    cv2.imshow("text only", text_only)
                    #cv.imshow("detected circles", graySegment)
                    cv.waitKey(0)
            segmentsDensity = segmentsDensity + 1



def main(argv):
    dataTrain = loadData('./train/', 'annotations')
    dataTest = loadData('./test/', 'annotations')
    dataTest[1].findCircles()
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

    gray = cv.medianBlur(gray, 5)

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