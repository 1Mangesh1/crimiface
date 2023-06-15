import cv2
import numpy as np
import os
import csv
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

# Define the label map
label_map = {}

# Iterate over the folders in the "criminal_images" directory
for folder in os.listdir('criminal_images'):
    if os.path.isdir(os.path.join('criminal_images', folder)):
        # Assign a label to each person based on the folder name
        label = len(label_map)
        label_map[folder] = label

# Load the criminal details from the CSV file
criminal_details = {}
with open('criminal_details.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        criminal_details[row['CID']] = row

# Prepare the training data
X_train = []
y_train = []

# Iterate over the criminal images
for folder in os.listdir('criminal_images'):
    if os.path.isdir(os.path.join('criminal_images', folder)):
        label = label_map[folder]
        for filename in os.listdir(os.path.join('criminal_images', folder)):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                img_path = os.path.join('criminal_images', folder, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # Iterate over the detected faces
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y + h, x:x + w]
                    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
                    face_roi_rgb = cv2.resize(face_roi_rgb, IMAGE_SIZE)
                    face_roi_rgb = preprocess_input(face_roi_rgb)
                    X_train.append(face_roi_rgb)
                    y_train.append(label)

# Convert the training data to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Perform one-hot encoding on the labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_train_one_hot = to_categorical(y_train_encoded)

# Split the data into training and validation sets
X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=42)

# Add a global average pooling layer and a fully connected layer on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(label_map), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators for data augmentation
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator()

# Define the checkpoint callback
checkpoint = ModelCheckpoint('criminal_recognition.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# Train the model
history = model.fit(train_datagen.flow(X_train, y_train_one_hot, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(X_train) // BATCH_SIZE, epochs=EPOCHS,
                    validation_data=val_datagen.flow(X_val, y_val_one_hot),
                    validation_steps=len(X_val) // BATCH_SIZE, callbacks=[checkpoint])

# Save the label map
np.save('label_map.npy', label_map)
