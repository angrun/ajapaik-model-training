# ProcessingImage(1, image_id, img_data, verdict_scene=verdict_scene_interior)

class DataQualityValidation:

    @staticmethod
    def prepare_report(all_feedbacks: dict, cleaned_up_feedbacks: list):
        print("Preparing the report")
        print(len(all_feedbacks))

        true_positive = 0
        true_negative = 0
        false_negative = 0
        false_positive = 0

        for feedback in all_feedbacks:
            category = feedback.split("_")[0]
            image_id = int(feedback.split("_")[1])

            # EXCLUDED FLOW
            if image_id in [el.image_id for el in cleaned_up_feedbacks]:
                if category == "exterior" and all_feedbacks[feedback]["interior"] > all_feedbacks[feedback]["exterior"]:
                    false_negative += all_feedbacks[feedback]["interior"]
                elif category == "exterior" and all_feedbacks[feedback]["interior"] <= all_feedbacks[feedback]["exterior"]:
                    true_negative += all_feedbacks[feedback]["interior"]
                elif category == "interior" and all_feedbacks[feedback]["exterior"] > all_feedbacks[feedback]["interior"]:
                    false_negative += all_feedbacks[feedback]["exterior"]
                elif category == "interior" and all_feedbacks[feedback]["exterior"] <= all_feedbacks[feedback]["interior"]:
                    true_negative += all_feedbacks[feedback]["exterior"]
                else:
                    print("NOT HANDLED")
                    print(feedback)
                    print(all_feedbacks[feedback])
            else:
                # No exclusion as interior was 0
                if category == "exterior" and all_feedbacks[feedback]["interior"] != 0:
                    false_positive += all_feedbacks[feedback]["interior"]

                elif category == "exterior" and all_feedbacks[feedback]["interior"] == 0:
                    true_positive += all_feedbacks[feedback]["exterior"]
                elif category == "interior" and all_feedbacks[feedback]["exterior"] != 0:
                    false_positive += all_feedbacks[feedback]["exterior"]
                elif category == "interior" and all_feedbacks[feedback]["exterior"] == 0:
                    true_positive += all_feedbacks[feedback]["interior"]

                else:
                    print("NOT HANDLED")
                    print(feedback)


        print("TRUE POSITIVE: " + str(true_positive))
        print("TRUE NEGATIVE: " + str(true_negative))
        print("FALSE POSITIVE: " + str(false_positive))
        print("FALSE NEGATIVE: " + str(false_negative))

    @staticmethod
    def get_image_id(image_category_id):
        return image_category_id.split("_")[1]
