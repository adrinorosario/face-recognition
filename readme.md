# Face Recognition Project

This project aims to recognize faces using OpenCV and the LBPH (Local Binary Patterns Histograms) face recognizer. The code detects faces in images, trains a model using those faces, and then uses the model to recognize faces in new images. Additionally, it includes functionality for recognizing eyes using a separate Haar Cascade classifier.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Testing the Model](#testing-the-model)
  - [Eye Recognition](#eye-recognition)
- [File Structure](#file-structure)
- [License](#license)

## Installation

To run this project, you'll need to install the following dependencies:

- OpenCV
- NumPy

You can install these packages using pip:

```sh
pip install numpy opencv-python
```

## Usage

### Data Preparation

1. Upload your images in the required structure.
2. Ensure that the images are organized in folders named after the person in the images.

### Training the Model

1. Detect faces in images and create a training set:

```python
create_training_set()
print(f"Length of features: {len(features)}")
print(f"Length of labels: {len(labels)}")
```

2. Train the LBPH face recognizer:

```python
face_recognizer = cv.face.LBPHFaceRecognizer_create()
np_features = np.array(features, dtype='object')
np_labels = np.array(labels)
face_recognizer.train(np_features, np_labels)
np.save('features.npy', np_features)
np.save('labels.npy', np_labels)
face_recognizer.save('faces_trained.yml')
```

### Testing the Model

1. Recognize a single image:

```python
def recognize_a_single_image(image_path):
  image = cv.imread(image_path)
  gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
  faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
  for(x,y,w,h) in faces_rect:
    face_region_of_interest = gray[y:y+h, x:x+w]
    label, accuracy = facesRecognizer.predict(face_region_of_interest)
  print(f"This is {people[label]} with an accuracy of {accuracy}%")
```

2. Recognize faces in a folder of images:

```python
def recognize_a_folder_of_images(folder_path):
  for images in os.listdir(folder_path):
    image_path = os.path.join(folder_path + "/", images)
    if check_extension(image_path) == 1:
      image_read = cv.imread(image_path)
      grayed = cv.cvtColor(image_read, cv.COLOR_BGR2GRAY)
      faces_rect = haar_cascade.detectMultiScale(grayed, 1.1, 4)
      for(x,y,w,h) in faces_rect:
        face_region_of_interest = grayed[y:y+h, x:x+w]
        label, accuracy = facesRecognizer.predict(face_region_of_interest)
      print(f"This is {people[label]} with an accuracy of {accuracy}%")
```

### Eye Recognition

1. Detect eyes in images and create a training set:

```python
def create_eye_training_set():
  for person in people:
    path = DIR + person
    label = people.index(person)
    for images in os.listdir(path):
      image_path = os.path.join(path, images)
      if check_extension(image_path) == 1:
        image = cv.imread(image_path)
        grayed = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        eye_rect = haar_eye_cascade.detectMultiScale(image, 1.1, 4)
        for(x,y,w,h) in eye_rect:
          region_of_interest = image[y:y+h, x:x+w]
          eye_features.append(region_of_interest)
          eye_labels.append(label)
      else:
        continue
create_eye_training_set()
print(f'Length of features: {len(eye_features)}')
print(f'Length of labels: {len(eye_labels)}')
```

## File Structure

- `harr_face_default.xml`: Haar Cascade classifier for face detection.
- `features.npy`, `labels.npy`: Numpy arrays storing the training features and labels.
- `faces_trained.yml`: Trained LBPH face recognizer model.
- `haar_eye.xml`: Haar Cascade classifier for eye detection.
- `/content/persons/`: Directory containing training images organized in subdirectories named after each person.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This documentation provides an overview of the project's purpose, installation steps, usage instructions, file structure, and license information to help users understand and utilize the code effectively.