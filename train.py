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

while True:
    index, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = fCascade.detectMultiScale(gray, 1.6, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 200), 8)
        faces = gray[y:y + h, x:x + w]
        faces_resize = cv2.resize(faces, (w, h))

        prediction = model.predict(faces_resize)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(prediction)

        if prediction[1] < 500:
            cv2.putText(image, f'{names[prediction[0]]} | {round(prediction[1], 6)}%', (x-20, y-20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
        else:
            cv2.putText(image, 'not recognized', (x-20, y-20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

