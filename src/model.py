from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization , LeakyReLU

def create_model(input_shape, num_classes):
     model = Sequential()
     model.add(Conv2D(32, (3, 3), input_shape=input_shape))
     model.add(LeakyReLU(alpha=0.1))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))
     model.add(Dropout(0.3))

    # Second convolutional block
     model.add(Conv2D(64, (3, 3)))
     model.add(LeakyReLU(alpha=0.1))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))
     model.add(Dropout(0.3))

    # Third convolutional block
     model.add(Conv2D(128, (3, 3)))
     model.add(LeakyReLU(alpha=0.1))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))
     model.add(Dropout(0.3))

    # Fourth convolutional block
     model.add(Conv2D(256, (3, 3)))
     model.add(LeakyReLU(alpha=0.1))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))
     model.add(Dropout(0.3))

    # Flatten and dense layers
     model.add(Flatten())
     model.add(Dense(512))
     model.add(LeakyReLU(alpha=0.1))
     model.add(BatchNormalization())
     model.add(Dropout(0.5))

    # Output layer for binary classification
     model.add(Dense(1, activation='sigmoid'))
     return model
