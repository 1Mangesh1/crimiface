import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv

# Create a folder to store the criminal images
criminal_images_folder = 'criminal_images'
os.makedirs(criminal_images_folder, exist_ok=True)

# Create a CSV file to store the criminal details
criminal_details_file = 'criminal_details.csv'
if not os.path.exists(criminal_details_file):
    with open(criminal_details_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CID', 'Name', 'DOB', 'Age', 'Last Location', 'Crimes'])

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
    capture_images(path)

    # Write the criminal details to the CSV file
    with open(criminal_details_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cid, name, dob, age, last_location, crimes])

    messagebox.showinfo("Success", "Data collection completed!")

def capture_images(path):
    video = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')
    profile_face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_profileface.xml')

    count = 0

    while count < 500:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            count += 1
            image_path = os.path.join(path, f'{count}.jpg')
            cv2.imwrite(image_path, frame[y:y + h, x:x + w])

            # Perform eye and smile detection within the detected face region
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            smiles = smile_cascade.detectMultiScale(roi_gray)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)

        profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in profile_faces:
            count += 1
            image_path = os.path.join(path, f'{count}.jpg')
            cv2.imwrite(image_path, frame[y:y + h, x:x + w])

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Create the Tkinter window
window = tk.Tk()
window.title("Data Collection")

# Create the labels and entry fields for data collection
cid_label = tk.Label(window, text="CID:")
cid_label.pack()

cid_entry = tk.Entry(window)
cid_entry.pack()

name_label = tk.Label(window, text="Name:")
name_label.pack()

name_entry = tk.Entry(window)
name_entry.pack()

dob_label = tk.Label(window, text="Date of Birth (DOB):")
dob_label.pack()

dob_entry = tk.Entry(window)
dob_entry.pack()

age_label = tk.Label(window, text="Age:")
age_label.pack()

age_entry = tk.Entry(window)
age_entry.pack()

location_label = tk.Label(window, text="Last Location:")
location_label.pack()

location_entry = tk.Entry(window)
location_entry.pack()

crimes_label = tk.Label(window, text="Crimes:")
crimes_label.pack()

crimes_entry = tk.Entry(window)
crimes_entry.pack()

# Create the start button for data collection
start_button = tk.Button(window, text="Start Data Collection", command=start_data_collection)
start_button.pack()

# Start the Tkinter event loop
window.mainloop()
