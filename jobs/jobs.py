import requests
import json

from service.image_processing_service import ProcessingService
from service.model.image_scene_model_prediction import ScenePrediction
from service.model.image_view_point_elevation_model_prediction import ViewPointElevationPrediction
from service.quality.data_quality_engine import DataQuality

URL = 'http://localhost:8000/'

IMAGE_AGGREGATION = "IMAGE AGGREGATION"
UNCATEGORIZED_IMAGES = "UNCATEGORIZED IMAGES"
SCENE = "SCENE"
VIEWPOINT_ELEVATION = "VIEWPOINT_ELEVATION"


def aggregate_and_retrain_model():
    response = requests.get(URL + 'object-categorization/aggregate-category-data')
    if response.status_code == 200:
        print(f"{IMAGE_AGGREGATION}: Data for retraining fetched successfully\n")

        images_ready_for_processing = ProcessingService.process_images_for_retraining(response.content)

        if not images_ready_for_processing:
            print(f"{IMAGE_AGGREGATION}: No new images available for retraining process\n")
        else:
            print(f"{IMAGE_AGGREGATION}: {len(images_ready_for_processing)} images taken for retraining")

            images_processed_through_data_quality_engine_view_point_elevation = \
                DataQuality.exclude_faulty_feedback_viewpoint_elevation_v3(images_ready_for_processing)

            images_processed_through_data_quality_engine_scene = \
                DataQuality.exclude_faulty_feedback_scene_v3(images_ready_for_processing)

            images_ready_for_processing_scene = images_processed_through_data_quality_engine_scene[0]
            images_ready_for_processing_viewpoint_elevation = \
                images_processed_through_data_quality_engine_view_point_elevation[0]
            images_excluded_scene = images_processed_through_data_quality_engine_scene[1]
            images_excluded_viewpoint_elevation = images_processed_through_data_quality_engine_view_point_elevation[1]

            print(
                f"{IMAGE_AGGREGATION} ({SCENE}): {len(images_excluded_scene)} feedbacks excluded")
            print(
                f"{IMAGE_AGGREGATION} ({SCENE}): {len(images_ready_for_processing_scene)} taken for retraining")
            print(
                f"{IMAGE_AGGREGATION} ({VIEWPOINT_ELEVATION}): {len(images_excluded_viewpoint_elevation)} feedbacks excluded")
            print(
                f"{IMAGE_AGGREGATION} ({VIEWPOINT_ELEVATION}): {len(images_ready_for_processing_viewpoint_elevation)} taken for retraining")

            if images_ready_for_processing_scene:
                ScenePrediction.retrain_model(images_ready_for_processing_scene)
            if images_ready_for_processing_viewpoint_elevation:
                ViewPointElevationPrediction.retrain_model(images_ready_for_processing_viewpoint_elevation)

    else:
        print(f"{IMAGE_AGGREGATION}: request failed with status code {response.status_code}")


def categorize_uncategorized_images():
    print(f"{UNCATEGORIZED_IMAGES}: Getting uncategorized images\n")

    response = requests.get(URL + 'object-categorization/get-uncategorized-images')
    if response.status_code == 200:
        response_data = json.loads(response.content.decode('utf-8'))
        data = response_data['data']

        images_ready_for_processing = ProcessingService.process(data)
        if not images_ready_for_processing:
            print(f"{UNCATEGORIZED_IMAGES}: no new uncategorized images available\n")

        else:
            print(
                f"{UNCATEGORIZED_IMAGES}: received {len(images_ready_for_processing)} images for category predictions\n")
            for image in images_ready_for_processing:
                scene_prediction = ScenePrediction.predict(image)
                view_point_elevation_prediction = ViewPointElevationPrediction.predict(image)
                image_to_send = ProcessingService.prepare_prediction_for_final_verdict(image, scene_prediction,
                                                                                       view_point_elevation_prediction)
                ProcessingService.post_model_predictions_to_result_table(image_to_send)
    else:
        print(f"{UNCATEGORIZED_IMAGES}: request failed with status code {response.status_code}")
