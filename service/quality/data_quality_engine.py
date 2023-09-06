from collections import defaultdict, Counter
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from service.image_processing_service import ProcessingImage


class DataQuality:

    # Version 1
    @staticmethod
    def exclude_faulty_feedback_v1(user_feedback):
        print("USER FEEDBACK")
        print(user_feedback)
        verdict_counts_per_image = defaultdict(Counter)

        for feedback in user_feedback:
            image_id = feedback.image_id
            verdict = feedback.verdict_scene
            verdict_counts_per_image[image_id][verdict] += 1

        print("===")
        print(verdict_counts_per_image)
        most_common_verdicts = {}
        for image_id, verdict_counts in verdict_counts_per_image.items():
            most_common_verdicts[image_id] = verdict_counts.most_common(1)[0][0]

        cleanup_feedback = []
        removed_feedback = []
        for feedback in user_feedback:
            if most_common_verdicts[feedback.image_id] == feedback.verdict_scene:
                cleanup_feedback.append(feedback)
            else:
                removed_feedback.append(feedback)

        print(f"DATA CLEAN UP PERFORMED, REMOVED {len(removed_feedback)}")
        print(f"DATA CLEAN UP PERFORMED, CONSIDERING {len(cleanup_feedback)}")
        return cleanup_feedback

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
        return cleaned_feedback