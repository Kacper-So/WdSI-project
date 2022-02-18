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
import itertools

# TODO Jakość kodu i raport (4/4)


# TODO Skuteczność klasyfikacji 0.0 (0/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0

# stderr:
# Traceback (most recent call last):
#   File "main.py", line 280, in <module>
#     main(sys.argv[1:])
#   File "main.py", line 263, in main
#     trainImages = loadImages(os.path.join(os.path.split(os.path.dirname(__file__))[0]), 'train', 'images') #trainImages = lista z nazwami obrazów w pliku train
#   File "main.py", line 18, in loadImages
#     for file in os.listdir(os.path.join(os.path.join(path, filename),filename2)):
# FileNotFoundError: [Errno 2] No such file or directory: 'train/images'

# TODO Skuteczność detekcji 0.0 (0/2)

# stderr:
# Traceback (most recent call last):
#   File "main.py", line 280, in <module>
#     main(sys.argv[1:])
#   File "main.py", line 263, in main
#     trainImages = loadImages(os.path.join(os.path.split(os.path.dirname(__file__))[0]), 'train', 'images') #trainImages = lista z nazwami obrazów w pliku train
#   File "main.py", line 18, in loadImages
#     for file in os.listdir(os.path.join(os.path.join(path, filename),filename2)):
# FileNotFoundError: [Errno 2] No such file or directory: 'train/images'

# TODO max(0, 0+0) = 0


#funkcja zwraca listę nazw obrazów znajdujących się w podanym pliku
def loadImages(path,filename, filename2):
    data = []
    for file in os.listdir(os.path.join(path, filename,filename2)):
        f = os.path.join(path, filename,filename2,file)
        if f.endswith(".png"):
            name = os.path.basename(f)
            data.append(name)
    return data

#funkcja na podstawie listy nazw obrazów odczytuje pliki xml (jeżeli istnieją) i tworzy obiekt typu ImageAnalizer. Funkcja zwraca listę obiektów typu ImageAnalizer
def loadData(path, imgNames):
    data = []
    for img in imgNames:
        name = os.path.splitext(img)[0]
        if os.path.isfile(os.path.join(path, 'annotations', name + '.xml')):
            f = os.path.join(path, 'annotations', os.path.splitext(img)[0]+'.xml')
            XMLtree = ET.parse(f)
            XMLroot = XMLtree.getroot()
            data.append(ImageAnalizer(path,f, name))
        else:
            data.append(ImageAnalizer(path, None, name))
    return data

#klasa zawierająca wszystkie informacje o obrazie jak i sam obraz oraz metodę pozwalającą na wykrycie obszarów w których potencjalnie może znajdować się znak
class ImageAnalizer:
    def __init__(self, path, f, name):
        self.name = name
        self.path = path
        self.XMLf = f
        self.img = cv2.imread(os.path.join(os.path.join(self.path, 'images'),self.name+'.png'), cv.IMREAD_COLOR)
        self.readXML()

    def readXML(self):
        if self.XMLf != None:
            self.XMLtree = ET.parse(self.XMLf)
            XMLroot = self.XMLtree.getroot()
            # TODO Lepiej uzyc funkcji "find" oraz "findall".
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
        return

    def findCircles(self):
        self.readXML()
        #cv.imshow("detected circles", self.img)
        #cv.waitKey(0)
        tempImg = self.img.copy()
        test = self.img.copy()
        #imageSegments = [[] for i in range(3)]
        imageSegments = []
        filtrationRange = [(0, 10, 20), (40, 255, 255),(140, 10, 20),(180, 255, 255)]
        hsv = cv2.cvtColor(tempImg, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, filtrationRange[0], filtrationRange[1])
        mask2 = cv2.inRange(hsv, filtrationRange[2], filtrationRange[3])
        mask = cv2.bitwise_or(mask1, mask2)
        croped = cv2.bitwise_and(tempImg, tempImg, mask=mask)

        segmentsDensity = 1
        for iter in range(1):
            dx = int(self.width / segmentsDensity)
            dy = int(self.height / segmentsDensity)
            for iterY in range(segmentsDensity):
                for iterX in range(segmentsDensity):
                    segment = croped.copy()
                    segment = segment[iterY*dy:iterY*dy+dy, iterX*dx:iterX*dx+dx]
                    graySegment = cv.cvtColor(segment, cv.COLOR_BGR2GRAY)
                    graySegment = cv.medianBlur(graySegment, 3)
                    #cv2.imshow('img', graySegment)
                    #cv2.waitKey(0)
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
                            imageSegments.append({'image': temp, 'name': self.name, 'ymin': y, 'ymax': y+h, 'xmin': x, 'xmax': x+w})
                    test = self.img.copy()
            segmentsDensity = segmentsDensity + 1
        for dict in imageSegments:
            dict.update({'n': len(imageSegments)})
        return imageSegments

#funkcja zwracająca słownik składający się z obrazów oraz ich classID odczytanych z pliku XML
def dataMerge(dataX):
    data = []
    for image in dataX:
        if image.XMLf != None:
            for iter in range(len(image.classID)):
                if image.classID[iter] == 'speedlimit':
                    temp = 1
                else:
                    temp = 0
                # TODO Przydalyby sie tez przyklady np. tla czy innych obiektow w klasie "other".
                data.append({'image': image.img[image.ymin[iter]: image.ymax[iter], image.xmin[iter]: image.xmax[iter]],'label': temp})
<<<<<<< HEAD
=======
        else:
            # TODO W tym przypadku zmienna "iter" bedzie niezainicjowana.
            data.append({'image': image.img[image.ymin[iter]: image.ymax[iter], image.xmin[iter]: image.xmax[iter]]})
>>>>>>> 65eab48ab3b372cb29f3c8d44b109bea92352317
    return data

#funkcja tworząca słownik potrzebny do uczenia maszynowego przy pomocy sift
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

#funkcja przetwarzająca dane, wyciągająca punkty kluczowe z obrazu przy pomocy sift
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

#funkcja tworząca randomForest
def train(data):
    clf = RandomForestClassifier(128)
    x_matrix = np.empty((1, 128))
    y_vector = []
    for sample in data:
        y_vector.append(sample['label'])
        x_matrix = np.vstack((x_matrix, sample['desc']))
    clf.fit(x_matrix[1:], y_vector)
    return clf

#funkcja klasyfikująca obrazy przy pomocy uczenia maszynowego
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

#funkcja spełniająca funkcjonalność classify
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
        for image in dataTest:
            if image.name == file:
                for j in range(n):
                    data.append({'image': image.img[bboxes[j][2]:bboxes[j][3], bboxes[j][0]:bboxes[j][1]]})
    return data

#funkcja wyświetlająca wyniki klasyfikacji
def displayResultsClassify(data):
    for img in data:
        if img['label_pred'] == 1:
            print('speedlimit')
        else:
            print('other')
    return

def detect(dataTest, rf):
    data = []
    for imgAn in dataTest:
        data.append(imgAn.findCircles())
    data = [j for temp2 in data for j in temp2]
    data = predict(rf, extractFeatures(data))
    data2 = []
    for dict in data:
<<<<<<< HEAD
        if dict['label_pred'] == 1:
            # TODO To moze nie dzialac prawidlowo, poniewaz rozmiar data bedzie sie zmienial.
            data2.append(dict)
=======
        if dict['label_pred'] != 1:
            # TODO To moze nie dzialac prawidlowo, poniewaz rozmiar data bedzie sie zmienial.
            del data[iter]
        iter = iter + 1
>>>>>>> 65eab48ab3b372cb29f3c8d44b109bea92352317
    iter = 0
    name = ''
    for dict in data2:
        if dict['name'] != name:
            iter = iter + 1
            name = dict['name']
    images = [[] for i in range(iter)]
    if data2 != []:
        iter = 0
        name = data2[0]['name']
        for dict in data2:
            images[iter].append(dict)
            if dict['name'] != name:
                iter = iter + 1
                name = dict['name']
    n = 0
    for img in images:
        if img != []:
            for dict in img:
                n = n + 1
            img[0]['n'] = n
            n = 0
    for img in images:
        if img != []:
            print(img[0]['name'])
            print(img[0]['n'])
            for segment in img:
                print(segment['xmin'], segment['xmax'], segment['ymin'], segment['ymax'], sep = ' ')
    return

def main(argv):
    command = input('')
    # TODO Zla sciezka.
    trainImages = loadImages(os.path.join(os.path.split(os.path.dirname(__file__))[0]), 'train', 'images') #trainImages = lista z nazwami obrazów w pliku train
    testImages = loadImages(os.path.join(os.path.split(os.path.dirname(__file__))[0]), 'test', 'images') #testImages = lista z nazwami obrazów w pliku test
    dataTrain = loadData(os.path.join(os.path.split(os.path.dirname(__file__))[0], 'train'), trainImages) #dataTrain = lista obiektów typu ImageAnalizer wygenerowana na podstawie nazw obrazów
    dataTest = loadData(os.path.join(os.path.split(os.path.dirname(__file__))[0], 'test'), testImages) #dataTest = lista obiektów typu ImageAnalizer wygenerowana na podstawie nazw obrazów
    dataTrain = dataMerge(dataTrain) #dataTrain = słownik składający się z obrazów oraz ich classID, generowany dla dataTrain z uwagi na pewność wstąienia pliku XML
                                     #oraz tego iż taki słownik będzie potrzebny tylko podczas uczenia
    learn(dataTrain) #generowanie słownika dla uczenia maszynowego
    dataTrain = extractFeatures(dataTrain) #dataTrain = słownik składający się z kluczowych punktów obrazów
    rf = train(dataTrain) #rf = randomForest stworzony na podstawie słownika składającego się z kluczowych punktów
    if command == 'classify':
        displayResultsClassify(predict(rf, extractFeatures(classify(dataTest))))
    if command == 'detect':
        detect(dataTest, rf)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])