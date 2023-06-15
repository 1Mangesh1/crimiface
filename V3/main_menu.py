import tkinter as tk
import subprocess
import sys

def add_face():
    # Run the data collection script
    subprocess.call([sys.executable, "datacollect.py"])

def train_faces():
    # Run the trainer script
    subprocess.call([sys.executable, "trainer.py"])

def find_criminals():
    # Run the recognition script
    subprocess.call([sys.executable, "recog.py"])

# Create the main window
window = tk.Tk()
window.title("Criminal Recognition System")

# Main Menu
main_menu_frame = tk.Frame(window)

add_face_button = tk.Button(main_menu_frame, text="Add Face", command=add_face)
add_face_button.pack()

train_faces_button = tk.Button(main_menu_frame, text="Train Faces", command=train_faces)
train_faces_button.pack()

find_criminals_button = tk.Button(main_menu_frame, text="Find Criminals", command=find_criminals)
find_criminals_button.pack()

main_menu_frame.pack()

# Start the Tkinter event loop
window.mainloop()
