import cv2
import os
import tkinter as tk
from tkinter import filedialog

# Load the trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read('trained_model.yml')

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

# Define the label map
label_map = {}

# Iterate over the folders in the "images" directory
for folder in os.listdir('images'):
    if os.path.isdir(os.path.join('images', folder)):
        # Assign a label to each person based on the folder name
        label = len(label_map)
        label_map[label] = folder

def separate_and_recognize_faces():
    # Open a file dialog to select the group photo
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Load the group photo
        group_image = cv2.imread(file_path)
        gray = cv2.cvtColor(group_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the group photo
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            # Recognize the face using the trained model
            label, confidence = model.predict(face_roi)

            if confidence < 100:
                recognized_person = label_map.get(label, 'Unknown')
            else:
                recognized_person = 'Unknown'

            # Draw a rectangle around the face and display the name
            cv2.rectangle(group_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(group_image, recognized_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Group Photo', group_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Create the main window
window = tk.Tk()
window.title("Face Separation and Recognition")

# Button to select the group photo and perform separation and recognition
select_button = tk.Button(window, text="Select Group Photo", command=separate_and_recognize_faces)
select_button.pack()

# Start the Tkinter event loop
window.mainloop()
