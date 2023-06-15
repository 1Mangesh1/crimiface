import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
model = load_model('V3\criminal_recognition.h5')

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('V3\data\haarcascades\haarcascade_frontalface_default.xml')

# Load the label map
label_map = np.load('V3\label_map.npy', allow_pickle=True).item()

def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        face_roi_rgb = cv2.resize(face_roi_rgb, (224, 224))
        face_roi_rgb = preprocess_input(face_roi_rgb)
        face_roi_input = np.expand_dims(face_roi_rgb, axis=0)

        # Recognize the face using the trained model
        prediction = model.predict(face_roi_input)
        predicted_label = np.argmax(prediction)
        confidence = prediction[0][predicted_label]

        if confidence > 0.5:
            recognized_person = list(label_map.keys())[list(label_map.values()).index(predicted_label)]
            messagebox.showinfo("Criminal Information", recognized_person)
        else:
            recognized_person = 'Unknown'

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Face Recognition', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def capture_image():
    # Open a file dialog to select the image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Load the image
        image = cv2.imread(file_path)

        # Recognize faces in the image
        recognize_faces(image)


# Create the main window
window = tk.Tk()
window.title("Face Recognition")

# Button to capture and recognize faces from an image
capture_button = tk.Button(window, text="Capture Image", command=capture_image)
capture_button.pack()

# Start the Tkinter event loop
window.mainloop()
