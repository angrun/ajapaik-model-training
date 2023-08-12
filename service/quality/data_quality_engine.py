from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataQuality:

    @staticmethod
    def exclude_faulty_feedback(feedback_data):
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
