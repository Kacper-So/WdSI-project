#!/usr/bin/env python

import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import os
import xml.etree.ElementTree as ET
import sys
import cv2 as cv

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
        self.readXML()

    def readXML(self):
        self.XMLtree = ET.parse(self.XMLf)
        XMLroot = self.XMLtree.getroot()
        self.name = XMLroot[1].text
        self.width = int(XMLroot[2][0].text)
        self.height = int(XMLroot[2][1].text)
        iter = 4
        self.xmin = []
        self.ymin = []
        self.xmax = []
        self.ymax = []
        self.classID = []
        while iter is not len(XMLroot):
            self.xmin.append(int(XMLroot[iter][5][0].text))
            self.ymin.append(int(XMLroot[iter][5][1].text))
            self.xmax.append(int(XMLroot[iter][5][2].text))
            self.ymax.append(int(XMLroot[iter][5][3].text))
            self.classID.append(XMLroot[iter][0].text)
            iter = iter + 1
        self.img = cv2.imread(os.path.join(os.path.join(self.path, 'images'),self.name), cv.IMREAD_COLOR)
        return

    def findCircles(self):
        self.readXML()
        cv.imshow("detected circles", self.img)
        cv.waitKey(0)
        tempImg = self.img.copy()
        test = self.img.copy()
        imageSegments = [[] for i in range(3)]
        for iterator in range(3):
            filtrationRanges = [[(0, 20, 20), (75, 255, 255),(105, 20, 20),(180, 255, 255)],
                                [(0, 10, 20), (85, 255, 255),(95, 10, 20),(180, 255, 255)],
                                [(0, 0, 0), (90, 255, 255),(90, 0,0),(180, 255, 255)]]
            hsv = cv2.cvtColor(tempImg, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, filtrationRanges[iterator][0], filtrationRanges[iterator][1])
            mask2 = cv2.inRange(hsv, filtrationRanges[iterator][2], filtrationRanges[iterator][3])
            mask = cv2.bitwise_or(mask1, mask2)
            croped = cv2.bitwise_and(tempImg, tempImg, mask=mask)

            segmentsDensity = 1
            for iter in range(2):
                dx = int(self.width / segmentsDensity)
                dy = int(self.height / segmentsDensity)
                for iterY in range(segmentsDensity):
                    for iterX in range(segmentsDensity):
                        segment = croped.copy()
                        segment = segment[iterY*dy:iterY*dy+dy, iterX*dx:iterX*dx+dx]
                        graySegment = cv.cvtColor(segment, cv.COLOR_BGR2GRAY)
                        graySegment = cv.medianBlur(graySegment, 3)
                        cv2.imshow('img', graySegment)
                        cv2.waitKey(0)
                        mser = cv2.MSER_create()
                        regions, bboxes = mser.detectRegions(graySegment)
                        for box in bboxes:
                            x, y, w, h = box
                            x = x + iterX * dx
                            y = y + iterY * dy
                            tolerance = int(0.05 * ((w + h) / 2))
                            if abs(w - h) < tolerance:
                                cv2.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 1)
                                temp = self.img[y : y + h, x : x + w].copy()
                                imageSegments[iterator].append(temp)
                        cv2.imshow('img2', test)
                        cv2.waitKey(0)
                        test = self.img.copy()
                segmentsDensity = segmentsDensity + 1
        return imageSegments

def dataMerge(dataX):
    data = []
    for image in dataX:
        image.readXML()
        for iter in range(len(image.classID)):
            if image.classID[iter] == 'speedlimit':
                temp = 1
            else:
                temp = 0
            data.append({'image': image.img[image.ymin[iter]: image.ymax[iter], image.xmin[iter]: image.xmax[iter]],'label': temp})
    return data

def learn(data):
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)
    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)
        if desc is not None:
            bow.add(desc)
    vocabulary = bow.cluster()
    np.save('voc.npy', vocabulary)
    return

def extractFeatures(data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        imgDes = bow.compute(sample['image'], kpts)
        if imgDes is not None:
            sample.update({'desc': imgDes})
        else:
            sample.update({'desc': np.zeros((1, 128))})
    return data

def train(data):
    clf = RandomForestClassifier(128)
    x_matrix = np.empty((1, 128))
    y_vector = []
    for sample in data:
        y_vector.append(sample['label'])
        x_matrix = np.vstack((x_matrix, sample['desc']))
    clf.fit(x_matrix[1:], y_vector)
    return clf

def predict(rf, data):
    for sample in data:
        sample.update({'label_pred': rf.predict(sample['desc'])[0]})
    return data

def evaluate(data):
    y_pred = []
    y_real = []
    for sample in data:
        y_pred.append(sample['label_pred'])
        y_real.append(sample['label'])

    confusion = confusion_matrix(y_real, y_pred)
    TP, FP, FN, TN = confusion.ravel()
    print(confusion)
    accuracy = 100 * ((TP + TN)/(TP + TN + FN + FP))
    print("accuracy =", round(accuracy, 2), "%")
    return

def classify(dataTest):
    data = []
    n_files = int(input(''))
    #print(n_files)
    for i in range(n_files):
        file = input('')
        #print(file)
        n = int(input(''))
        #print(n)
        bboxes = []
        for j in range(n):
            bboxes.append(input(''))
            bboxes[j] = [int(x) for x in bboxes[j].split()]
        #print(bboxes)
        for img in dataTest:
            if img.name == file:
                for j in range(n):
                    data.append({'image': img.img[bboxes[j][2]:bboxes[j][3], bboxes[j][0]:bboxes[j][1]]})
    return data

def displayResultsClassify(data):
    for img in data:
        #print(img['label_pred'])
        if img['label_pred'] == 1:
            print('speedlimit')
        else:
            print('other')

def main(argv):
    command = input('')
    #Training
    dataTrain = loadData('./train/', 'annotations')
    dataTest = loadData('./test/', 'annotations')
    dataTrain = dataMerge(dataTrain)
    #dataTest = dataMerge(dataTest)
    #print('lerning')
    learn(dataTrain)
    #print('extracting train features')
    dataTrain = extractFeatures(dataTrain)
    #print('training')
    rf = train(dataTrain)
    #print('extracting test features')
    #dataTest = extractFeatures(dataTest)
    #print('testing')
    #dataTest = predict(rf, dataTest)
    #evaluate(dataTest)
    if command == 'classify':
        displayResultsClassify(predict(rf, extractFeatures(classify(dataTest))))
    if command == 'detect':



    return 0


if __name__ == "__main__":
    main(sys.argv[1:])