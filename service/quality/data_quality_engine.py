import urllib.request
from collections import defaultdict, Counter
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import numpy as np

# from keras.utils import img_to_array

from io import BytesIO
from PIL import Image
from keras.utils import img_to_array

from service.image_processing_service import ProcessingImage
from service.model.image_scene_model_prediction import ScenePrediction


# from service.model.image_scene_model_prediction import ScenePrediction
CORRECT_EXTERIOR = 0

class DataQuality:


    # Version 1
    @staticmethod
    def exclude_faulty_feedback_v1(user_feedback):
        verdict_counts_per_image = defaultdict(Counter)

        for feedback in user_feedback:
            image_id = feedback.image_id
            verdict = feedback.verdict_scene
            verdict_counts_per_image[image_id][verdict] += 1

        most_common_verdicts = {}
        for image_id, verdict_counts in verdict_counts_per_image.items():
            most_common_verdicts[image_id] = verdict_counts.most_common(1)[0][0]

        print("MOST COMMON VERDICTS")
        print(most_common_verdicts)

        cleanup_feedback = []
        removed_feedback = []
        for feedback in user_feedback:
            if most_common_verdicts[feedback.image_id] == feedback.verdict_scene:
                print("===")
                print(most_common_verdicts[feedback.image_id])
                print(feedback.verdict_scene)
                cleanup_feedback.append(feedback)
            else:
                print("REMOVED FEEDBACK: " + str(feedback))
                removed_feedback.append(feedback)

        print(f"DATA CLEAN UP PERFORMED, REMOVED {len(removed_feedback)}")
        print(f"DATA CLEAN UP PERFORMED, CONSIDERING {len(cleanup_feedback)}")
        return cleanup_feedback, removed_feedback

    # Version 2
    @staticmethod
    def exclude_faulty_feedback_v2(feedback_data):
        user_data = {}
        for entry in feedback_data:
            user_id = entry.user_id
            if user_id not in user_data:
                user_data[user_id] = {'verdict_scene_0_count': 0, 'verdict_scene_1_count': 0}
            if entry.verdict_scene == 0:
                user_data[user_id]['verdict_scene_0_count'] += 1
            else:
                user_data[user_id]['verdict_scene_1_count'] += 1

        # Create features for anomaly detection
        features = []
        for user_id, counts in user_data.items():
            ratio = counts['verdict_scene_1_count'] / (
                    counts['verdict_scene_0_count'] + counts['verdict_scene_1_count'])
            features.append([ratio])

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Apply Isolation Forest for anomaly detection
        clf = IsolationForest(contamination=0.05)  # Adjust contamination based on your dataset
        clf.fit(scaled_features)

        # Identify users with potentially faulty feedback
        faulty_users = set()
        for i, prediction in enumerate(clf.predict(scaled_features)):
            if prediction == -1:
                faulty_users.add(list(user_data.keys())[i])

        faulty_feedback = []
        cleaned_feedback = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_lines = [f"Report generated at: {timestamp}\n\n"]

        for entry in feedback_data:
            if entry.user_id not in faulty_users:
                cleaned_feedback.append(entry)
            else:
                faulty_feedback.append(entry)
                anomaly_value = clf.decision_function([scaled_features[i]])[0]  # Get anomaly value
                report_lines.append(f"Excluded data: {entry}, Anomaly value: {anomaly_value}\n")

        report_filename = f"reports/exclusion_report_{timestamp.replace(' ', '_').replace(':', '-')}.txt"
        with open(report_filename, 'w') as report_file:
            report_file.writelines(report_lines)

        print(f"aggregate-category-data: Finished data quality engine, removed {len(faulty_feedback)} faulty records")

        return cleaned_feedback, faulty_feedback

    @staticmethod
    def exclude_faulty_feedback_v3(feedback_data):


        m = {"CORRECT_EXTERIOR": 0, "incorrect_exterior": 0, "correct_interior": 0, "incorrect_interior": 0}
        incorrect_exclusion_for_exterior = 0
        correct_exclusion_for_exterior = 0
        incorrect_exclusion_for_interior = 0
        correct_exclusion_for_interior = 0

        no_exclusion_correct = 0
        no_exclusion_wrong = 0

        faulty_feedbacks = []
        cleanup_data = []
        for feedback in feedback_data:
            model_prediction = DataQuality.get_image_prediction(feedback.image_id, m)
            if feedback.verdict_scene != model_prediction:
                if feedback.image_id <= 1019 and feedback.verdict_scene == 1:
                   incorrect_exclusion_for_exterior += 1
                if feedback.image_id <= 1019 and feedback.verdict_scene == 0:
                    correct_exclusion_for_exterior += 1
                if feedback.image_id > 1019 and feedback.verdict_scene == 0:
                    incorrect_exclusion_for_interior += 1
                if feedback.image_id > 1019 and feedback.verdict_scene == 1:
                    correct_exclusion_for_interior += 1
                faulty_feedbacks.append(feedback)
            else:
                if feedback.image_id <= 1019 and feedback.verdict_scene == 0:
                    no_exclusion_wrong += 1
                if feedback.image_id > 1019 and feedback.verdict_scene == 1:
                    no_exclusion_wrong += 1
                else:
                    no_exclusion_correct += 1
                cleanup_data.append(feedback)

        print("incorrect_exclusion_for_exterior: " + str(incorrect_exclusion_for_exterior))
        print("correct_exclusion_for_exterior: " + str(correct_exclusion_for_exterior))
        print("incorrect_exclusion_for_interior: " + str(incorrect_exclusion_for_interior))
        print("correct_exclusion_for_interior: " + str(correct_exclusion_for_interior))
        print("no_exclusion_correct: " + str(no_exclusion_correct))
        print("no_exclusion_wrong: " + str(no_exclusion_wrong))
        print("CORRECT_EXTERIOR: " + str(m))

        print(f"aggregate-category-data: Finished data quality engine, removed {len(faulty_feedbacks)} faulty records")
        return cleanup_data, faulty_feedbacks

    @staticmethod
    def get_image_prediction(image_id, m):
        model_prediction = 1
        IMG_WIDTH, IMG_HEIGHT = 224, 224
        THUMB_URL = "http://localhost:8000/"
        THUMB_PREFIX = "photo-thumb/"
        verdict_scene_interior = 0
        url = f"{THUMB_URL}{THUMB_PREFIX}{image_id}"
        with urllib.request.urlopen(url) as url_response:
            img_data = url_response.read()

            image = Image.open(BytesIO(img_data))

            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image_array = img_to_array(image)

            # Preprocess your image
            image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Classify your image
            class_probabilities = ScenePrediction.model.predict(image_array)[0][0]

            # 0 - interior, 1 = exterior
            predictions = {0: 1 - class_probabilities, 1: class_probabilities}
            if predictions[0] > predictions[1]:
                model_prediction = 0

            if image_id <= 1019:
                if model_prediction == 1:
                    m["CORRECT_EXTERIOR"] = m["CORRECT_EXTERIOR"] + 1
                else:
                    m["incorrect_exterior"] = m["incorrect_exterior"] + 1
            if image_id > 1019:
                if model_prediction == 0:
                    m["correct_interior"] = m["correct_interior"] + 1
                else:
                    m["incorrect_interior"] = m["incorrect_interior"] + 1
            return model_prediction





