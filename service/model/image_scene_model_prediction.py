import io
import os
import tensorflow
from tensorflow.keras.utils import to_categorical

from keras import Sequential
from keras.layers import Flatten, Dense
from keras.models import load_model
from keras.utils import img_to_array

from io import BytesIO
from PIL import Image

from service.image_processing_service import ProcessingImage

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SCENE_DIR = 'resources/scene'
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

model = None


class ScenePrediction:
    model = None
    model_path = 'saved_model.h5'

    def __init__(self):
        if ScenePrediction.model is None:
            self.model_start_up()

    @staticmethod
    def model_start_up():
        if os.path.isfile(ScenePrediction.model_path):
            print("loading model from cache")
            ScenePrediction.model = load_model(ScenePrediction.model_path)
        else:

            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True
            )
            train_generator = train_datagen.flow_from_directory(
                SCENE_DIR,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                classes=['interior', 'exterior']
            )

            val_datagen = ImageDataGenerator(rescale=1. / 255)
            val_generator = val_datagen.flow_from_directory(
                SCENE_DIR,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                classes=['interior', 'exterior']
            )

            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            base_model.trainable = False

            model = Sequential([
                base_model,
                Flatten(),
                BatchNormalization(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            callbacks = [
                EarlyStopping(patience=3, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.1, patience=2)
            ]

            model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=callbacks)
            ScenePrediction.model = model
            ScenePrediction.model.save(ScenePrediction.model_path)

    @staticmethod
    def predict(img_data: ProcessingImage):
        image = Image.open(BytesIO(img_data.image_for_processing))

        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = img_to_array(image)

        # Preprocess your image
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Classify your image
        class_probabilities = ScenePrediction.model.predict(image_array)[0][0]

        # 0 - interior, 1 = exterior
        predictions = {0: 1 - class_probabilities, 1: class_probabilities}
        return predictions

    @staticmethod
    def retrain_model(processed_images):
        new_images = []  # List to store preprocessed images
        new_labels = []  # List to store corresponding labels

        for image in processed_images:
            image_data = image.image_for_processing
            label = image.verdict_scene

            # Convert the image data to a PIL image
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Resize the image to the desired dimensions (IMG_WIDTH, IMG_HEIGHT)
            pil_image = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))

            # Convert the PIL image to numpy array
            np_image = np.array(pil_image)

            # Normalize the image data
            np_image = np_image.astype('float32') / 255.0

            new_images.append(np_image)
            new_labels.append(label)

        new_images = np.array(new_images)
        new_labels = np.array(new_labels)

        num_classes = 2  # Assuming there are 2 classes: interior and exterior

        # Load the saved model
        model = tensorflow.keras.models.load_model(ScenePrediction.model_path)

        # Replace the last layer with a new layer for the desired number of classes
        model.pop()
        model.add(Dense(num_classes, activation='softmax', name='output'))

        # Retrain the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(new_images, to_categorical(new_labels, num_classes), epochs=20,
                  validation_split=0.2)  # You can adjust the validation split

        # Save the retrained model
        model.save(ScenePrediction.model_path)
        print("Model is retrained")

    @staticmethod
    def determine_validation(new_images):
        min_samples = 10  # Minimum number of samples required for validation

        # Calculate the validation split based on the number of samples
        if len(new_images) >= min_samples:
            validation_split = min_samples / len(new_images)
        else:
            validation_split = 0.0  # No validation set if there are fewer samples than the minimum threshold
        return validation_split
