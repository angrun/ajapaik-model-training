import io
import os
import tensorflow
from keras.applications import ResNet50
from keras.callbacks import LearningRateScheduler
# from keras.src.callbacks import LearningRateScheduler
# from keras.src.callbacks import LearningRateScheduler
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

VIEW_DIR = 'resources/view_point_elevation'
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

model = None


class ViewPointElevationPrediction:
    model = None
    model_path = 'saved_model_view.h5'

    def __init__(self):
        if ViewPointElevationPrediction.model is None:
            self.model_start_up()


    @staticmethod
    def model_start_up():
        if os.path.isfile(ViewPointElevationPrediction.model_path):
            print("VIEW: Loading model from cache")
            ViewPointElevationPrediction.model = load_model(ViewPointElevationPrediction.model_path)
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
                VIEW_DIR,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode='categorical',  # Change to categorical
                classes=['ground', 'raised', 'aerial']  # Modify to your category names
            )

            val_datagen = ImageDataGenerator(rescale=1. / 255)
            val_generator = val_datagen.flow_from_directory(
                VIEW_DIR,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode='categorical',  # Change to categorical
                classes=['ground', 'raised', 'aerial']  # Modify to your category names
            )

            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            base_model.trainable = False


            model = Sequential([
                base_model,
                Flatten(),
                BatchNormalization(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(3, activation='softmax')  # Change to 3 output units and softmax activation
            ])

            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                          metrics=['accuracy'])  # Change loss function

            callbacks = [
                EarlyStopping(patience=3, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.1, patience=2)
            ]

            model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=callbacks)
            ViewPointElevationPrediction.model = model
            ViewPointElevationPrediction.model.save(ViewPointElevationPrediction.model_path)

    @staticmethod
    def predict(img_data: ProcessingImage):
        image = Image.open(BytesIO(img_data.image_for_processing))

        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = img_to_array(image)

        # Preprocess your image
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Classify your image
        class_probabilities = ViewPointElevationPrediction.model.predict(image_array)[0]

        probabilities_sum = np.sum(np.exp(class_probabilities))
        normalized_probabilities = np.exp(class_probabilities) / probabilities_sum

        # Map class probabilities to category labels
        category_labels = ['ground', 'raised', 'aerial']
        predictions = {category_labels[i]: normalized_probabilities[i] for i in range(len(normalized_probabilities))}

        return predictions

    @staticmethod
    def retrain_model(processed_images):
        images = []  # List to store preprocessed images
        verdicts = []  # List to store corresponding labels

        for image in processed_images:
            image_data = image.image_for_processing
            label = image.verdict_view_point_elevation

            # Convert the image data to a PIL image
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Resize the image to the desired dimensions (IMG_WIDTH, IMG_HEIGHT)
            pil_image = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))

            # Convert the PIL image to a numpy array
            np_image = np.array(pil_image)

            # Normalize the image data
            np_image = np_image.astype('float32') / 255.0

            images.append(np_image)
            verdicts.append(label)

        # Load the existing model or use the one you've already trained.
        model = ViewPointElevationPrediction.model
        print(f"MODEL IS: {model}")

        # One-hot encode the labels
        verdicts = to_categorical(verdicts, num_classes=3)

        # Use data augmentation during training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        # Create a data generator with additional augmentation
        train_generator = train_datagen.flow(
            np.array(images),
            verdicts,  # Use one-hot encoded labels
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # Fine-tune the existing model with the user feedback data.
        optimizer = Adam(learning_rate=0.0001)

        # Implement learning rate schedule to reduce learning rate during training
        def learning_rate_schedule(epoch, lr):
            if epoch < 20:
                return lr
            else:
                return lr * 0.1

        learning_rate_scheduler = LearningRateScheduler(learning_rate_schedule)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Create a separate validation data generator
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_generator = val_datagen.flow(
            np.array(images),
            verdicts,  # Use one-hot encoded labels
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # Implement early stopping and learning rate reduction on plateau
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),  # Increased patience
            ReduceLROnPlateau(factor=0.1, patience=5)  # Increased patience
        ]

        # Train the model
        # with user feedback data using separate training and validation generators
        model.fit(
            train_generator,
            epochs=50,
            validation_data=val_generator,  # Use separate validation data generator
            callbacks=callbacks + [learning_rate_scheduler],  # Include learning rate schedule
            shuffle=True  # Shuffle training data
        )

        # Save the retrained model.
        ViewPointElevationPrediction.model = model
        ViewPointElevationPrediction.model.save(ViewPointElevationPrediction.model_path)
        print("VIEW: Model retraining complete.")

    @staticmethod
    def determine_validation(new_images):
        min_samples = 10  # Minimum number of samples required for validation

        # Calculate the validation split based on the number of samples
        if len(new_images) >= min_samples:
            validation_split = min_samples / len(new_images)
        else:
            validation_split = 0.0  # No validation set if there are fewer samples than the minimum threshold
        return validation_split
