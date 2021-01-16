import cv2
import numpy
import os


face_file = 'haarcascade_frontalface_default.xml'
dataset = 'datasets'
subdata = 'sub'

path = os.path.join(dataset, subdata)
if not os.path.isdir(path):
    os.mkdir(path)

w, h = (500, 500)
fCascade = cv2.CascadeClassifier(face_file)
camera = cv2.VideoCapture(0)

count = 1
while count < 80:
    index, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = fCascade.detectMultiScale(gray, 1.6, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 200), 4)
        faces = gray[y:y+h, x:x+w]
        faces_resize = cv2.resize(faces, (w, h))
        cv2.imwrite(f'{path}/{count}.png', faces_resize)
    count += 1

    cv2.imshow('opencv', image)
    key = cv2.waitKey(10)
    if key == 27:
        break