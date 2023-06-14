
import cv2
import os
import tkinter as tk
from tkinter import filedialog

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

def separate_faces():
    # Open a file dialog to select the group photo
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Load the group photo
        group_image = cv2.imread(file_path)
        gray = cv2.cvtColor(group_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the group photo
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Create a directory to save the individual faces
        output_dir = 'separated_faces'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate over the detected faces
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = group_image[y:y + h, x:x + w]
            output_path = os.path.join(output_dir, f'face_{i}.jpg')
            cv2.imwrite(output_path, face_roi)

        # Display the result
        cv2.imshow('Group Photo', group_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Create the main window
window = tk.Tk()
window.title("Face Separation from Group Photo")

# Button to select the group photo
select_button = tk.Button(window, text="Select Group Photo", command=separate_faces)
select_button.pack()

# Start the Tkinter event loop
window.mainloop()
