import requests
import logging
import json

from service.image_processing_service import ProcessingService
from service.model.image_scene_model_prediction import ScenePrediction
from service.model.image_view_point_elevation_model_prediction import ViewPointElevationPrediction
from service.quality.data_quality_engine import DataQuality
from service.test.image_processing_service_test import ProcessingServiceTest
from service.validation.data_quality_validation_service import DataQualityValidation

logger = logging.getLogger(__name__)
URL = 'http://localhost:8000/'

IMAGE_AGGREGATION = "IMAGE AGGREGATION"
UNCATEGORIZED_IMAGES = "UNCATEGORIZED IMAGES"


def aggregate_and_retrain_model():
    response = requests.get(URL + 'object-categorization/aggregate-category-data')
    if response.status_code == 200:
        print(f"{IMAGE_AGGREGATION}: Data for retraining fetched successfully\n")

        # images_ready_for_processing = ProcessingService.process_images_for_retraining(response.content)
        processed_images = ProcessingServiceTest.process_images_for_retraining_v3()  # Mock data

        images_ready_for_processing = processed_images[0]
        collected_report = processed_images[1]

        if not images_ready_for_processing:
            print(f"{IMAGE_AGGREGATION}: No new images available for retraining process\n")
        else:
            print(f"{IMAGE_AGGREGATION}: {len(images_ready_for_processing)} images taken for retraining")

            images_processed_through_data_quality_engine = \
                DataQuality.exclude_faulty_feedback_v3(images_ready_for_processing)

            images_ready_for_processing = images_processed_through_data_quality_engine[0]
            images_excluded = images_processed_through_data_quality_engine[1]

            # REPORT
            DataQualityValidation.prepare_report(collected_report, images_excluded)

            print(
                f"{IMAGE_AGGREGATION}: {len(images_excluded)} feedbacks excluded")
            print(
                f"{IMAGE_AGGREGATION}: {len(images_ready_for_processing)} feedbacks taken for retraining after quality engine")
            ScenePrediction.retrain_model(images_ready_for_processing)

    else:
        print(f"{IMAGE_AGGREGATION}: request failed with status code {response.status_code}")


def categorize_uncategorized_images():
    print("get-uncategorized-images: Getting uncategorized images\n")

    response = requests.get(URL + 'object-categorization/get-uncategorized-images')
    if response.status_code == 200:
        logger.info(f"{UNCATEGORIZED_IMAGES}: data fetched successfully")
        response_data = json.loads(response.content.decode('utf-8'))
        data = response_data['data']  # [photo_id, user_id, photo_name]

        images_ready_for_processing = ProcessingService.process(data)
        if not images_ready_for_processing:
            print(f"{UNCATEGORIZED_IMAGES}: no new uncategorized images available\n")

        else:
            print(
                f"{UNCATEGORIZED_IMAGES}: received {len(images_ready_for_processing)} images for category predictions\n")
            for image in images_ready_for_processing:
                # scene_prediction = ScenePrediction.predict(image)
                view_point_elevation_prediction = ViewPointElevationPrediction.predict(image)

                image_to_send = ProcessingService.prepare_prediction_for_final_verdict(image, None, view_point_elevation_prediction)
                print("===")
                print(image_to_send)
                ProcessingService.post_model_predictions_to_result_table(image_to_send)
    else:
        print(f"{UNCATEGORIZED_IMAGES}: request failed with status code {response.status_code}")
