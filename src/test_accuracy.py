import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Function to load the model
def load_model(model_path):
    return joblib.load(model_path)

# Function to preprocess test data
def preprocess_test_data(test_dir, target_size=(256, 256), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return test_generator

# Function to evaluate the model
def evaluate_model(model, test_generator):
    # Make predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

if __name__ == "__main__":
    # Define paths
    model_path = 'C:/projects/BrainTumorPrediction/trained_model.pkl'  # Path to your saved model
    test_dir = 'C:/projects/BrainTumorPrediction/data/resized_test'  # Path to your test directory

    # Load model
    model = load_model(model_path)

    # Preprocess test data
    test_generator = preprocess_test_data(test_dir)

    # Evaluate the model
    evaluate_model(model, test_generator)
