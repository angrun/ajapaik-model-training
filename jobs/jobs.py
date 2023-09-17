import requests
import logging
import json

from service.image_processing_service import ProcessingService
from service.model.image_scene_model_prediction import ScenePrediction
from service.quality.data_quality_engine import DataQuality
from service.test.image_processing_service_test import ProcessingServiceTest

logger = logging.getLogger(__name__)
URL = 'http://localhost:8000/'

IMAGE_AGGREGATION = "IMAGE AGGREGATION"
UNCATEGORIZED_IMAGES = "UNCATEGORIZED IMAGES"


def aggregate_and_retrain_model():
    response = requests.get(URL + 'object-categorization/aggregate-category-data')
    if response.status_code == 200:
        print(f"aggregate-category-data: Data for retraining fetched successfully\n")

        # images_ready_for_processing = ProcessingService.process_images_for_retraining(response.content)
        images_ready_for_processing = ProcessingServiceTest.process_images_for_retraining()
        if not images_ready_for_processing:
            print("aggregate-category-data: No new images available for retraining process\n")
        else:
            # Filter out fraud predictions
            # print(f"aggregate-category-data: {len(images_ready_for_processing)} images taken for retraining")
            # images_ready_for_processing = DataQuality.exclude_faulty_feedback_v1(images_ready_for_processing)
            # print(
            #     f"aggregate-category-data {len(images_ready_for_processing)} images taken for retraining after quality engine")
            ScenePrediction.retrain_model(images_ready_for_processing)

    else:
        print(f"Request failed with status code {response.status_code}")
        logger.info(f"Data fetching failed with status code {response.status_code}")


def categorize_uncategorized_images():
    print("get-uncategorized-images: Getting uncategorized images\n")

    response = requests.get(URL + 'object-categorization/get-uncategorized-images')
    if response.status_code == 200:
        logger.info(f"{UNCATEGORIZED_IMAGES}: Data fetched successfully")
        response_data = json.loads(response.content.decode('utf-8'))
        data = response_data['data']  # [photo_id, user_id, photo_name]

        images_ready_for_processing = ProcessingService.process(data)
        if not images_ready_for_processing:
            print("get-uncategorized-images: No new uncategorized images available\n")

        else:
            print(
                f"get-uncategorized-images: Received {len(images_ready_for_processing)} images for category predictions\n")
            for image in images_ready_for_processing:
                scene_prediction = ScenePrediction.predict(image)
                image_to_send = ProcessingService.prepare_prediction_for_final_verdict(image, scene_prediction)
                ProcessingService.batch_to_result_table(image_to_send)
    else:
        print(f"Request failed with status code {response.status_code}")
        logger.info(f"Ung fetching failed with status code {response.status_code}")


def represent(data):
    input_data_str = data.decode('utf-8')

    data = json.loads(input_data_str)

    alternative_data = data['data']['alternative_category_data']

    print("ID\tSCENE\tVIEW_POINT\tPROPOSER\tDATE")

    for entry in alternative_data:
        pk = entry['pk']
        scene = entry['fields']['scene_alternation'] if entry['fields']['scene_alternation'] is not None else '-'
        viewpoint = entry['fields']['viewpoint_elevation_alternation'] if entry['fields'][
                                                                              'viewpoint_elevation_alternation'] is not None else '-'
        proposer = entry['fields']['proposer']
        created = entry['fields']['created']
        print(f"{pk}\t\t{scene}\t\t{viewpoint}\t\t{proposer}\t\t\t{created}")
