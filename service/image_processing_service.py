import json
import urllib.request
import requests

THUMB_URL = "http://localhost:8000/"
THUMB_PREFIX = "photo-thumb/"
RESULT_PREFIX = "object-categorization/publish-picture-category-result/"
HEADERS = {"Content-Type": "application/json"}
DECISION_RATE = 0.5


class ProcessingService:

    @staticmethod
    def process(image_payload):
        processed_images = []
        for image in image_payload:
            image_id = image[0]
            user_id = image[1]
            image_url = image[2]
            url = f"{THUMB_URL}{THUMB_PREFIX}{image_id}"
            with urllib.request.urlopen(url) as url_response:
                img_data = url_response.read()
                processed_images.append(ProcessingImage(user_id, image_id, img_data))
        return processed_images

    @staticmethod
    def process_images_for_retraining(data):
        input_data_str = data.decode('utf-8')
        data = json.loads(input_data_str)

        processed_images = []
        if data['data'] != {}:
            alternative_data = data['data']['alternative_category_data']

            for entry in alternative_data:
                user_id = entry['fields']['proposer']
                image_id = entry['fields']['photo']
                scene_alternative = entry['fields']['scene_alternation']
                url = f"{THUMB_URL}{THUMB_PREFIX}{image_id}"
                with urllib.request.urlopen(url) as url_response:
                    img_data = url_response.read()
                    processed_images.append(
                        ProcessingImage(user_id, image_id, img_data, verdict_scene=scene_alternative))
        return processed_images

    @staticmethod
    def prepare_prediction_for_final_verdict(processed_image, prediction_result):
        filtered_d = {k: v for k, v in prediction_result.items() if v > DECISION_RATE}
        if filtered_d:
            processed_image.verdict_scene = max(filtered_d, key=filtered_d.get)
        return processed_image

    # TODO: to do it in batch mode
    @staticmethod
    def batch_to_result_table(processed_image):
        payload = {
            "photo_id": processed_image.image_id,
        }
        if processed_image.verdict_scene is not None:
            payload["verdict_scene"] = processed_image.verdict_scene

        if processed_image.verdict_view_point_elevation is not None:
            payload["verdict_view_point_elevation"] = processed_image.verdict_view_point_elevation

        response = requests.post(f"{THUMB_URL}{RESULT_PREFIX}", json=payload, headers=HEADERS)

        if response.status_code == 200:
            print("Success!")
        else:
            print(f"Error: {response.status_code} - {response.text}")


class ProcessingImage:
    def __init__(self, user_id, image_id, image_for_processing, verdict_scene=None, verdict_view_point_elevation=None):
        self.user_id = user_id
        self.image_id = image_id
        self.image_for_processing = image_for_processing
        self.verdict_scene = verdict_scene
        self.verdict_view_point_elevation = verdict_view_point_elevation

    def __str__(self):
        return f'[user_id: {self.user_id}, image_id: {self.image_id}, verdict_scene: {self.verdict_scene}]'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, ProcessingImage):
            return (
                    self.user_id == other.user_id and
                    self.image_id == other.image_id and
                    self.image_for_processing == other.image_for_processing and
                    self.verdict_scene == other.verdict_scene and
                    self.verdict_view_point_elevation == other.verdict_view_point_elevation
            )
        return False
