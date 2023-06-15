import cv2
import numpy as np
import os
import csv
import tkinter as tk
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from tkinter import messagebox

# Create a folder to store the criminal images
criminal_images_folder = 'criminal_images'
os.makedirs(criminal_images_folder, exist_ok=True)

# Create a CSV file to store the criminal details
criminal_details_file = 'criminal_details.csv'
if not os.path.exists(criminal_details_file):
    with open(criminal_details_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CID', 'Name', 'DOB', 'Age', 'Last Location', 'Crimes'])

# Load the Haar cascade 500
# classifier for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

# Create the data generator for augmentation
data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                   shear_range=0.15, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

def start_data_collection():
    cid = cid_entry.get()
    name = name_entry.get()
    dob = dob_entry.get()
    age = age_entry.get()
    last_location = location_entry.get()
    crimes = crimes_entry.get()

    if cid == '' or name == '' or dob == '' or age == '' or last_location == '' or crimes == '':
        messagebox.showerror("Error", "Please fill in all the fields.")
        return

    path = os.path.join(criminal_images_folder, cid)
    if os.path.exists(path):
        messagebox.showerror("Error", "CID already taken. Enter a different CID.")
        return

    os.makedirs(path)

    # Capture images for data collection
    capture_images(path)

    # Write the criminal details to the CSV file
    with open(criminal_details_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cid, name, dob, age, last_location, crimes])

    messagebox.showinfo("Success", "Data collection completed!")

def capture_images(path):
    # Open the webcam
    video = cv2.VideoCapture(0)

    # Set the image count and maximum limit
    count = 0
    max_images = 500

    while count < max_images:
        ret, frame = video.read()

        # Apply data augmentation
        frame = np.expand_dims(frame, axis=0)
        augmented_frame = next(data_generator.flow(frame, batch_size=1))[0].astype(np.uint8)

        # Convert frame to grayscale
        gray = cv2.cvtColor(augmented_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in faces:
            count += 1
            image_path = os.path.join(path, f'{count}.jpg')
            cv2.imwrite(image_path, augmented_frame[y:y + h, x:x + w])

        cv2.imshow("Capture", augmented_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Create the Tkinter window
window = tk.Tk()
window.title("Data Collection")

# Create labels
cid_label = tk.Label(window, text="CID:")
name_label = tk.Label(window, text="Name:")
dob_label = tk.Label(window, text="DOB:")
age_label = tk.Label(window, text="Age:")
location_label = tk.Label(window, text="Last Location:")
crimes_label = tk.Label(window, text="Crimes:")

# Create entry fields
cid_entry = tk.Entry(window)
name_entry = tk.Entry(window)
dob_entry = tk.Entry(window)
age_entry = tk.Entry(window)
location_entry = tk.Entry(window)
crimes_entry = tk.Entry(window)

# Create start button
start_button = tk.Button(window, text="Start Data Collection", command=start_data_collection)

# Place the labels, entry fields, and start button in the window
cid_label.grid(row=0, column=0)
name_label.grid(row=1, column=0)
dob_label.grid(row=2, column=0)
age_label.grid(row=3, column=0)
location_label.grid(row=4, column=0)
crimes_label.grid(row=5, column=0)

cid_entry.grid(row=0, column=1)
name_entry.grid(row=1, column=1)
dob_entry.grid(row=2, column=1)
age_entry.grid(row=3, column=1)
location_entry.grid(row=4, column=1)
crimes_entry.grid(row=5, column=1)

start_button.grid(row=6, column=0, columnspan=2)

# Start the Tkinter event loop
window.mainloop()
