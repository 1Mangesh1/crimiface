# Criminal Recognition System

This project aims to develop a Criminal Recognition System that can detect and recognize criminals from a group photo. The system utilizes face recognition techniques to match the faces in the photo with the known criminals in the database.

## Version 1: Basic Face Recognition

In the first version, the system performs basic face recognition by comparing the faces in the group photo with the images stored in the criminal database. The system displays the names of the identified criminals based on the folder names in the "criminal_images" directory.

To run Version 1:
1. Ensure that you have the necessary dependencies installed.
2. Run the `login.py` script to login to the system.
3. After successful login, run the `recog.py` script.
4. Select a group photo to analyze.
5. The system will display the recognized criminals' names in the photo, if any.

## Version 2: Detailed Criminal Information

In the second version, the system not only recognizes the criminals but also provides detailed information about them. The system reads the criminal details from a CSV file and displays additional information such as name, date of birth, age, last known location, and criminal records.

To run Version 2:
1. Ensure that you have the necessary dependencies installed.
2. Run the `login.py` script to login to the system.
3. After successful login, run the `recog.py` script.
4. Select a group photo to analyze.
5. The system will display the recognized criminals' names and their detailed information, if available.

## Version 3: Deep Learning Face Recognition

In the third version, the system incorporates deep learning techniques for face recognition. It uses the VGG16 model and the DeepFace library to extract facial features from the images and trains a neural network model for improved accuracy.

To run Version 3:
1. Ensure that you have the necessary dependencies installed, including the DeepFace library.
2. Run the `login.py` script to login to the system.
3. After successful login, run the `trainer.py` script to train the face recognition model.
4. Run the `recog.py` script.
5. Select a group photo to analyze.
6. The system will utilize the trained model to recognize and provide information about the criminals in the photo.

## Folder Structure

- `criminal_images/`: This folder contains subfolders named after each criminal's unique identifier (CID). Each subfolder contains images of the respective criminal.
- `criminal_details.csv`: This CSV file stores the detailed information of the criminals, including their CID, name, date of birth, age, last known location, and criminal records.

## Dependencies

The project relies on the following dependencies:
- OpenCV
- NumPy
- tkinter (for GUI)
- DeepFace (for Version 3)

Please ensure that these dependencies are installed before running the project.

## License

This project is licensed under the [MIT License](LICENSE).
