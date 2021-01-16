import cv2
import numpy
import os

face_file = 'haarcascade_frontalface_default.xml'
dataset = 'datasets'
images, labels, names, id = [], [], {}, 0

for subdirs, dirs, files in os.walk(dataset):
    for subdir in dirs:
        names[id] = subdir
        subpath = os.path.join(dataset, subdir)
        for filename in os.listdir(subpath):
            path = f'{subpath}/{filename}'
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
w, h = (500, 500)
images, labels = [numpy.array(lis) for lis in [images, labels]]

print(dir(cv2.face))
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

fCascade = cv2.CascadeClassifier(face_file)
camera = cv2.VideoCapture(0)