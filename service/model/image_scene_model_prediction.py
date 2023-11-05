import io
import os

from keras import Sequential
from keras.layers import Flatten, Dense
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
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
IMAGE_AGGREGATION = "IMAGE AGGREGATION"
UNCATEGORIZED_IMAGES = "UNCATEGORIZED IMAGES"
SCENE = "SCENE"

model = None


class ScenePrediction:
    model = None
    model_path = 'saved_model_scene.h5'

    def __init__(self):
        if ScenePrediction.model is None:
            self.model_start_up()

    @staticmethod
    def model_start_up():
        if os.path.isfile(ScenePrediction.model_path):
            print(f"{UNCATEGORIZED_IMAGES} ({SCENE}): loading model from cache")
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

        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        class_probabilities = ScenePrediction.model.predict(image_array)[0][0]

        # 0 - interior, 1 = exterior
        predictions = {0: 1 - class_probabilities, 1: class_probabilities}
        return predictions

    @staticmethod
    def retrain_model(processed_images):
        images = []
        verdicts = []

        for image in processed_images:
            image_data = image.image_for_processing
            label = image.verdict_scene

            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

            pil_image = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))

            np_image = np.array(pil_image)
            np_image = np_image.astype('float32') / 255.0

            images.append(np_image)
            verdicts.append(label)

        model = ScenePrediction.model

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
        train_generator = train_datagen.flow(
            np.array(images),
            np.array(verdicts),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        optimizer = Adam(learning_rate=0.0001)

        def learning_rate_schedule(epoch, lr):
            if epoch < 20:
                return lr
            else:
                return lr * 0.1

        learning_rate_scheduler = LearningRateScheduler(learning_rate_schedule)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_generator = val_datagen.flow(
            np.array(images),
            np.array(verdicts),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=5)
        ]

        model.fit(
            train_generator,
            epochs=50,
            validation_data=val_generator,
            callbacks=callbacks + [learning_rate_scheduler],
            shuffle=True
        )
        model.save(ScenePrediction.model_path)
        print(f"{IMAGE_AGGREGATION} ({SCENE}): Model retraining complete.")
