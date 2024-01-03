import urllib.request

import numpy as np

from io import BytesIO
from PIL import Image
from keras.utils import img_to_array

from service.model.image_scene_model_prediction import ScenePrediction
from service.model.image_view_point_elevation_model_prediction import ViewPointElevationPrediction

IMG_WIDTH, IMG_HEIGHT = 224, 224
THUMB_URL = "http://localhost:8000/"
THUMB_PREFIX = "photo-thumb/"


class DataQuality:

    @staticmethod
    def exclude_faulty_feedback_scene_v3(feedback_data):
        faulty_feedbacks = []
        cleanup_data = []
        for feedback in feedback_data:
            model_prediction = DataQuality.get_image_prediction_scene(feedback.image_id)
            if feedback.verdict_scene != model_prediction:
                faulty_feedbacks.append(feedback)
            else:
                cleanup_data.append(feedback)
        return cleanup_data, faulty_feedbacks

    @staticmethod
    def get_image_prediction_scene(image_id):
        model_prediction = 1
        url = f"{THUMB_URL}{THUMB_PREFIX}{image_id}"
        with urllib.request.urlopen(url) as url_response:
            img_data = url_response.read()

            image = Image.open(BytesIO(img_data))

            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image_array = img_to_array(image)

            image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            class_probabilities = ScenePrediction.model.predict(image_array)[0][0]

            predictions = {0: 1 - class_probabilities, 1: class_probabilities}
            if predictions[0] > predictions[1]:
                model_prediction = 0
            return model_prediction

    @staticmethod
    def exclude_faulty_feedback_viewpoint_elevation_v3(feedback_data):
        faulty_feedbacks = []
        cleanup_data = []
        for feedback in feedback_data:
            model_prediction = DataQuality.get_image_prediction_viewpoint_elevation(feedback.image_id)
            if feedback.verdict_view_point_elevation != model_prediction:
                faulty_feedbacks.append(feedback)
            else:
                cleanup_data.append(feedback)
        return cleanup_data, faulty_feedbacks

    @staticmethod
    def get_image_prediction_viewpoint_elevation(image_id):
        url = f"{THUMB_URL}{THUMB_PREFIX}{image_id}"
        with urllib.request.urlopen(url) as url_response:
            img_data = url_response.read()

            image = Image.open(BytesIO(img_data))

            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image_array = img_to_array(image)

            image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            class_probabilities = ViewPointElevationPrediction.model.predict(image_array)[0]
            probabilities_sum = np.sum(np.exp(class_probabilities))
            normalized_probabilities = np.exp(class_probabilities) / probabilities_sum

            category_labels = ['ground', 'raised', 'aerial']
            predictions = {category_labels[i]: normalized_probabilities[i] for i in
                           range(len(normalized_probabilities))}
            view_verdict = max(predictions, key=predictions.get)
            category_labels = {'ground': 0, 'raised': 1, 'aerial': 2}
            return category_labels[view_verdict]
