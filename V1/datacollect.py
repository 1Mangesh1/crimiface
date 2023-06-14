import cv2
import os
import tkinter as tk

def start_data_collection():
    name = name_entry.get()
    path = 'images/' + name.lower()

    if os.path.exists(path):
        message_label.config(text="Name already taken. Enter a different name.")
    else:
        os.makedirs(path)
        data_collection_frame.pack_forget()
        video_capture_frame.pack()
        name_label.config(text="Collecting data for: " + name)
        capture_images()

def capture_images():
    global count

    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for cascade in face_cascades:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for x, y, w, h in faces:
            count += 1
            name = './images/' + name_entry.get().lower() + '/' + str(count) + '.jpg'
            print("Creating Images........." + name)
            cv2.imwrite(name, frame[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1)

    if count <= 500:
        window.after(10, capture_images)
    else:
        video_capture_frame.pack_forget()
        completion_frame.pack()

# Initialize Tkinter window
window = tk.Tk()
window.title("Data Collection")

# Create frames
data_collection_frame = tk.Frame(window)
video_capture_frame = tk.Frame(window)
completion_frame = tk.Frame(window)

# Data Collection Frame
name_label = tk.Label(data_collection_frame, text="Enter the name of the criminal:")
name_label.pack()

name_entry = tk.Entry(data_collection_frame)
name_entry.pack()

start_button = tk.Button(data_collection_frame, text="Start Data Collection", command=start_data_collection)
start_button.pack()

message_label = tk.Label(data_collection_frame, fg="red")
message_label.pack()

# Video Capture Frame
video = cv2.VideoCapture(0)

cascade_files = [
    'data/haarcascades/haarcascade_frontalface_default.xml',
    'data/haarcascades/haarcascade_frontalface_alt_tree.xml',
    'data/haarcascades/haarcascade_frontalface_alt.xml',
    'data/haarcascades/haarcascade_frontalface_alt2.xml',
    'data/haarcascades/haarcascade_profileface.xml',
    'data/haarcascades/haarcascade_smile.xml',
]

face_cascades = []
for cascade_file in cascade_files:
    cascade = cv2.CascadeClassifier(cascade_file)
    face_cascades.append(cascade)

count = 0

# Completion Frame
completion_label = tk.Label(completion_frame, text="Data collection completed!")

completion_label.pack()

# Pack initial frame

data_collection_frame.pack()

# Start the Tkinter event loop
window.mainloop()
