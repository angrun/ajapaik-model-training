# ProcessingImage(1, image_id, img_data, verdict_scene=verdict_scene_interior)

class DataQualityValidation:

    @staticmethod
    def prepare_report(all_feedbacks: list, cleaned_up_feedbacks: list):
        print("Preparing the report")
        print(all_feedbacks)
