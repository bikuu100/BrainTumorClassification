

# Brain Tumor Classification

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Web Application](#web-application)
- [AWS Deployment](#aws-deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project focuses on the classification of brain tumors using machine learning techniques. The goal is to accurately classify brain tumor images into their respective categories and provide a web interface for users to upload and classify images.

## Dataset
The dataset used for this project consists of MRI images of brain tumors. Each image is labeled with a category indicating the type of brain tumor. The dataset can be downloaded from [link to dataset].

## Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow (or any other deep learning framework you prefer)
- flask
- boto3 (for AWS)

You can install the required libraries using:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow flask boto3
```

## Exploratory Data Analysis (EDA)
In the EDA phase, we analyze the dataset to gain insights into the data distribution, detect anomalies, and understand relationships between variables. The steps include:
1. **Loading the data**: Importing the dataset using Pandas.
2. **Data visualization**: Plotting the distribution of classes, sample images, and other relevant features.
3. **Statistical analysis**: Descriptive statistics to summarize the central tendency, dispersion, and shape of the dataset’s distribution.
4. **Correlation analysis**: Understanding the relationship between different features.

### Example Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('path/to/your/dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Plot the distribution of classes
sns.countplot(x='label', data=data)
plt.title('Distribution of Brain Tumor Types')
plt.show()

# Display sample images
sample_images = data.sample(5)
for index, row in sample_images.iterrows():
    img = plt.imread(row['image_path'])
    plt.imshow(img, cmap='gray')
    plt.title(row['label'])
    plt.show()
```

## Data Preprocessing
Preprocessing steps to prepare the data for modeling:
1. **Data cleaning**: Handling missing values, removing duplicates.
2. **Data augmentation**: Applying transformations to increase the diversity of the training set.
3. **Normalization**: Scaling pixel values to a range of 0-1.
4. **Splitting the data**: Dividing the data into training, validation, and test sets.

### Example Code
```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                             width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

train_generator = datagen.flow_from_dataframe(train_data, x_col='image_path', y_col='label', class_mode='categorical')
val_generator = datagen.flow_from_dataframe(val_data, x_col='image_path', y_col='label', class_mode='categorical')
test_generator = datagen.flow_from_dataframe(test_data, x_col='image_path', y_col='label', class_mode='categorical', shuffle=False)
```

## Modeling
In this phase, we build and train machine learning models to classify brain tumors. We use different models and compare their performance.

### Example Code
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=20)
```

## Evaluation
Evaluate the model's performance on the test set using metrics such as accuracy, precision, recall, and F1-score.

### Example Code
```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Predict on the test set
predictions = model.predict(test_generator)
predictions = np.argmax(predictions, axis=1)
true_labels = test_generator.classes

# Classification report
print(classification_report(true_labels, predictions, target_names=test_generator.class_indices.keys()))

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

## Visualization
Visualizing the training process and the performance of the model helps in understanding and interpreting the results.

### Example Code
```python
# Plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

## Web Application
We developed a Flask web application that allows users to upload MRI images and get predictions on brain tumor classification.

### Flask Application Structure
```
/app
    /static
        - styles.css
    /templates
        - index.html
        - result.html
    app.py
    model.py
```

### Flask Application Code

**app.py**
```python
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('path/to/your/model.h5')

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_names = ['Class1', 'Class2', 'Class3']  # Replace with actual class names
    return class_names[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            file_path = f"static/{file.filename}"
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('result.html', prediction=prediction, image_path=file_path)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
```

**index.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Brain Tumor Classification</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
</head>
<body>
    <h1>Brain Tumor Classification</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
</body>
</html>
```

**result.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prediction Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <p>The uploaded image is classified as: {{ prediction }}</p>
    <img src="{{ url_for('static', filename=image_path) }}" alt="Uploaded Image">
</body>
</html>
```

### Running the Flask App Locally
To run the Flask app locally, execute the following command:
