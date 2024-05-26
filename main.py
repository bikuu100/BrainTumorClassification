import os
import sys
sys.path.append('src')
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dropout
from preprocessing import preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import create_model
from loggers import setup_logger
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from exception import DataPreprocessingError, ModelCreationError, ModelEvaluationError
import joblib


logger = setup_logger('main', 'main.log')

if __name__ == "__main__":
    try:
        train_dir = 'C:/projects/BrainTumorPrediction/data/resized_train'
        test_dir = "C:/projects/BrainTumorPrediction/data/resized_test"
        class_labels = ["no", "yes"]
        train_generator, validation_generator, test_generator = preprocess(train_dir, test_dir)

        input_shape = train_generator.image_shape
        num_classes = len(class_labels)

        model = create_model(input_shape, num_classes)
        initial_learning_rate = 0.001
        lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )
        optimizer = Adam(learning_rate=lr_schedule)
      
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  
        

        history = model.fit(train_generator, epochs=100, validation_data=validation_generator,
                       )

        

        joblib.dump(model, 'trained_model.pkl')

    except (DataPreprocessingError, ModelCreationError, ModelEvaluationError) as e:
        logger.error(f"An error occurred: {str(e)}")

        