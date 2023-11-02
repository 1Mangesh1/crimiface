import cv2
import numpy as np
import os
import csv


face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')


training_data = []
labels = []
label_map = {}  


with open('criminal_details.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        cid = row[0]
        name = row[1]

        
        label = len(label_map)
        label_map[cid] = label

        
        person_folder = os.path.join('criminal_images', cid)
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            if not os.path.isfile(image_path):
                continue

            
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                training_data.append(face_roi)
                labels.append(label)


training_data = np.array(training_data, dtype=object)
labels = np.array(labels)


model = cv2.face.LBPHFaceRecognizer_create()
model.train(training_data, labels)


model.save('trained_model.yml')
