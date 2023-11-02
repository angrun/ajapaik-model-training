import urllib

from service.image_processing_service import ProcessingImage
import urllib.request
import random


# TODO: mock data used for validation purposes, to be removed

def handle_processed_image(collections, category, image_id, verdict):
    item = category + "_" + str(image_id)
    if item not in collections.keys():
        collections[item] = {"feedback_count": 0, "exterior": 0, "interior": 0}
        collections[item]["feedback_count"] = collections[item]["feedback_count"] + 1
        collections[item][verdict] = collections[item][verdict] + 1
    else:
        collections[item]["feedback_count"] = collections[item]["feedback_count"] + 1
        collections[item][verdict] = collections[item][verdict] + 1


class ProcessingServiceTest:

    @staticmethod
    def process_images_for_retraining():
        THUMB_URL = "http://localhost:8000/"
        THUMB_PREFIX = "photo-thumb/"
        image_id = 33
        verdict_scene_interior = 0
        url = f"{THUMB_URL}{THUMB_PREFIX}{image_id}"
        with urllib.request.urlopen(url) as url_response:
            img_data = url_response.read()

        return [
            ProcessingImage(1, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(2, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(3, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(4, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(5, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(6, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(7, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(8, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(9, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(10, image_id, img_data, verdict_scene=verdict_scene_interior),
            ProcessingImage(11, image_id, img_data, verdict_scene=1),
            ProcessingImage(12, image_id, img_data, verdict_scene=1),
            ProcessingImage(13, image_id, img_data, verdict_scene=1),
            ProcessingImage(14, image_id, img_data, verdict_scene=1),
            ProcessingImage(15, image_id, img_data, verdict_scene=1),
            ProcessingImage(16, image_id, img_data, verdict_scene=1),
            ProcessingImage(17, image_id, img_data, verdict_scene=1),
            ProcessingImage(18, image_id, img_data, verdict_scene=1),
            ProcessingImage(19, image_id, img_data, verdict_scene=1),
            ProcessingImage(20, image_id, img_data, verdict_scene=1),
            ProcessingImage(21, image_id, img_data, verdict_scene=1),
            ProcessingImage(22, image_id, img_data, verdict_scene=1),
            ProcessingImage(23, image_id, img_data, verdict_scene=1),
            ProcessingImage(24, image_id, img_data, verdict_scene=1),
            ProcessingImage(25, image_id, img_data, verdict_scene=1),
            ProcessingImage(26, image_id, img_data, verdict_scene=1),
            ProcessingImage(27, image_id, img_data, verdict_scene=1),
            ProcessingImage(28, image_id, img_data, verdict_scene=1),
            ProcessingImage(29, image_id, img_data, verdict_scene=1),
            ProcessingImage(30, image_id, img_data, verdict_scene=1),
            ProcessingImage(31, image_id, img_data, verdict_scene=1),
            ProcessingImage(32, image_id, img_data, verdict_scene=1),
            ProcessingImage(33, image_id, img_data, verdict_scene=1),
            ProcessingImage(34, image_id, img_data, verdict_scene=1),
            ProcessingImage(35, image_id, img_data, verdict_scene=1),
            ProcessingImage(36, image_id, img_data, verdict_scene=1),
            ProcessingImage(37, image_id, img_data, verdict_scene=1),
            ProcessingImage(38, image_id, img_data, verdict_scene=1),
            ProcessingImage(39, image_id, img_data, verdict_scene=1),
            ProcessingImage(40, image_id, img_data, verdict_scene=1),
            ProcessingImage(41, image_id, img_data, verdict_scene=1),
            ProcessingImage(42, image_id, img_data, verdict_scene=1),
            ProcessingImage(43, image_id, img_data, verdict_scene=1),
            ProcessingImage(44, image_id, img_data, verdict_scene=1),
            ProcessingImage(45, image_id, img_data, verdict_scene=1),
            ProcessingImage(46, image_id, img_data, verdict_scene=1),
            ProcessingImage(47, image_id, img_data, verdict_scene=1),
            ProcessingImage(48, image_id, img_data, verdict_scene=1),
            ProcessingImage(49, image_id, img_data, verdict_scene=1),
            ProcessingImage(50, image_id, img_data, verdict_scene=1),
            ProcessingImage(51, image_id, img_data, verdict_scene=1),
            ProcessingImage(52, image_id, img_data, verdict_scene=1),
            ProcessingImage(53, image_id, img_data, verdict_scene=1),
            ProcessingImage(54, image_id, img_data, verdict_scene=1),
            ProcessingImage(55, image_id, img_data, verdict_scene=1),
            ProcessingImage(56, image_id, img_data, verdict_scene=1),
            ProcessingImage(57, image_id, img_data, verdict_scene=1),
            ProcessingImage(58, image_id, img_data, verdict_scene=1),
            ProcessingImage(59, image_id, img_data, verdict_scene=1),
            ProcessingImage(60, image_id, img_data, verdict_scene=1),
            ProcessingImage(61, image_id, img_data, verdict_scene=1),
            ProcessingImage(62, image_id, img_data, verdict_scene=1),
            ProcessingImage(63, image_id, img_data, verdict_scene=1),
            ProcessingImage(64, image_id, img_data, verdict_scene=1),
            ProcessingImage(65, image_id, img_data, verdict_scene=1),
            ProcessingImage(66, image_id, img_data, verdict_scene=1),
            ProcessingImage(67, image_id, img_data, verdict_scene=1),
            ProcessingImage(68, image_id, img_data, verdict_scene=1),
            ProcessingImage(69, image_id, img_data, verdict_scene=1),
            ProcessingImage(70, image_id, img_data, verdict_scene=1),
            ProcessingImage(71, image_id, img_data, verdict_scene=1),
            ProcessingImage(72, image_id, img_data, verdict_scene=1),
            ProcessingImage(73, image_id, img_data, verdict_scene=1),
            ProcessingImage(74, image_id, img_data, verdict_scene=1),
            ProcessingImage(75, image_id, img_data, verdict_scene=1),
            ProcessingImage(76, image_id, img_data, verdict_scene=1),
            ProcessingImage(77, image_id, img_data, verdict_scene=1),
            ProcessingImage(78, image_id, img_data, verdict_scene=1),
            ProcessingImage(79, image_id, img_data, verdict_scene=1),
            ProcessingImage(80, image_id, img_data, verdict_scene=1),
            ProcessingImage(81, image_id, img_data, verdict_scene=1),
            ProcessingImage(82, image_id, img_data, verdict_scene=1),
            ProcessingImage(83, image_id, img_data, verdict_scene=1),
            ProcessingImage(84, image_id, img_data, verdict_scene=1),
            ProcessingImage(85, image_id, img_data, verdict_scene=1),
            ProcessingImage(86, image_id, img_data, verdict_scene=1),
            ProcessingImage(87, image_id, img_data, verdict_scene=1),
            ProcessingImage(88, image_id, img_data, verdict_scene=1),
            ProcessingImage(89, image_id, img_data, verdict_scene=1),
            ProcessingImage(90, image_id, img_data, verdict_scene=1),
            ProcessingImage(91, image_id, img_data, verdict_scene=1),
            ProcessingImage(92, image_id, img_data, verdict_scene=1),
            ProcessingImage(93, image_id, img_data, verdict_scene=1),
            ProcessingImage(94, image_id, img_data, verdict_scene=1),
            ProcessingImage(95, image_id, img_data, verdict_scene=1),
            ProcessingImage(96, image_id, img_data, verdict_scene=1),
            ProcessingImage(97, image_id, img_data, verdict_scene=1),
            ProcessingImage(98, image_id, img_data, verdict_scene=1),
            ProcessingImage(99, image_id, img_data, verdict_scene=1),
            ProcessingImage(100, image_id, img_data, verdict_scene=1)
        ]

    @staticmethod
    def process_images_for_retraining_v2():
        uniqueness_check = []
        THUMB_URL = "http://localhost:8000/"
        THUMB_PREFIX = "photo-thumb/"
        image_id_exterior_from = 1
        image_id_exterior_to = 1019
        image_id_interior_from = 1020
        image_id_interior_to = 2038
        user_id = 1

        result = []

        exterior = {"exterior_correct": 0, "exterior_wrong": 0}
        interior = {"interior_correct": 0, "interior_wrong": 0}

        while image_id_exterior_from != image_id_exterior_to:
            try:
                url = f"{THUMB_URL}{THUMB_PREFIX}{image_id_exterior_from}"
                with urllib.request.urlopen(url) as url_response:
                    img_data = url_response.read()
                    verdict_fo_exterior = random.randint(0, 1)
                    if verdict_fo_exterior == 0:
                        exterior["exterior_wrong"] = exterior["exterior_wrong"] + 1
                    else:
                        exterior["exterior_correct"] = exterior["exterior_correct"] + 1
                    result.append(
                        ProcessingImage(user_id, image_id_exterior_from, img_data, verdict_scene=verdict_fo_exterior))
            except Exception as e:
                pass
            image_id_exterior_from += 1
            user_id += 1

        print(exterior)

        while image_id_interior_from != image_id_interior_to:
            try:
                url = f"{THUMB_URL}{THUMB_PREFIX}{image_id_interior_from}"
                with urllib.request.urlopen(url) as url_response:
                    img_data = url_response.read()
                    verdict_fo_interior = random.randint(0, 1)
                    if verdict_fo_interior == 1:
                        interior["interior_wrong"] = interior["interior_wrong"] + 1
                    else:
                        interior["interior_correct"] = interior["interior_correct"] + 1
                    result.append(
                        ProcessingImage(user_id, image_id_interior_from, img_data, verdict_scene=verdict_fo_interior))
            except Exception:
                print("Caught exception")
            image_id_interior_from += 1
            user_id += 1

        print(interior)

        return result

    @staticmethod
    def process_images_for_retraining_v3():
        uniqueness_check = []

        THUMB_URL = "http://localhost:8000/"
        THUMB_PREFIX = "photo-thumb/"
        image_id_exterior_from = random.randint(1, 1019)
        image_id_interior_from = random.randint(1020, 2038)
        user_id = random.randint(1, 100)
        counter = 0

        result = []
        collections = {}

        exterior = {"exterior_correct": 0, "exterior_wrong": 0}
        interior = {"interior_correct": 0, "interior_wrong": 0}

        while counter != 750:
            image_id_exterior_from = random.randint(1, 1019)
            user_id = random.randint(1, 100)
            try:
                # Ensure only 1 feedback per image from a user
                if [user_id, image_id_exterior_from] in uniqueness_check:
                    print("EXTERIOR: Uniqueness break")
                    continue
                else:
                    uniqueness_check.append([user_id, image_id_exterior_from])

                url = f"{THUMB_URL}{THUMB_PREFIX}{image_id_exterior_from}"
                with urllib.request.urlopen(url) as url_response:
                    img_data = url_response.read()
                    verdict_fo_exterior = random.randint(0, 1)

                    handle_processed_image(collections, "exterior", image_id_exterior_from,
                                           "exterior" if verdict_fo_exterior == 1 else "interior")

                    if verdict_fo_exterior == 0:
                        exterior["exterior_wrong"] = exterior["exterior_wrong"] + 1
                    else:
                        exterior["exterior_correct"] = exterior["exterior_correct"] + 1
                    result.append(
                        ProcessingImage(user_id, image_id_exterior_from, img_data, verdict_scene=verdict_fo_exterior))
            except Exception as e:
                print(e)
                print("EXCEPTION")
            counter += 1

        print(exterior)
        counter = 0

        while counter != 750:
            image_id_interior_from = random.randint(1020, 2038)
            user_id = random.randint(1, 100)
            try:
                if [user_id, image_id_interior_from] in uniqueness_check:
                    print("INTERIOR: Uniqueness break")
                    continue
                else:
                    uniqueness_check.append([user_id, image_id_interior_from])

                url = f"{THUMB_URL}{THUMB_PREFIX}{image_id_interior_from}"
                with urllib.request.urlopen(url) as url_response:
                    img_data = url_response.read()
                    verdict_to_interior = random.randint(0, 1)

                    handle_processed_image(collections, "interior", image_id_interior_from,
                                           "exterior" if verdict_to_interior == 1 else "interior")

                    if verdict_to_interior == 1:
                        interior["interior_wrong"] = interior["interior_wrong"] + 1
                    else:
                        interior["interior_correct"] = interior["interior_correct"] + 1
                    result.append(
                        ProcessingImage(user_id, image_id_interior_from, img_data, verdict_scene=verdict_to_interior))
            except Exception:
                print("Caught exception")
            counter += 1

        print(interior)

        return result, collections

    @staticmethod
    def process_images_for_retraining_view_v3():
        uniqueness_check = []

        THUMB_URL = "http://localhost:8000/"
        THUMB_PREFIX = "photo-thumb/"
        image_id_aerial_from = random.randint(1, 1019)
        image_id_interior_from = random.randint(1020, 2038)
        user_id = random.randint(1, 100)
        counter = 0

        result = []
        collections = {}

        ground = {"ground_correct": 0, "ground_wrong": 0}
        raised = {"raised_correct": 0, "raised_wrong": 0}
        aerial = {"aerial_correct": 0, "aerial_wrong": 0}

        while counter != 500:
            image_id_aerial_from = random.randint(2039, 3057)
            user_id = random.randint(1, 100)
            try:
                # Ensure only 1 feedback per image from a user
                if [user_id, image_id_aerial_from] in uniqueness_check:
                    print("AERIAL: Uniqueness break")
                    continue
                else:
                    uniqueness_check.append([user_id, image_id_aerial_from])

                url = f"{THUMB_URL}{THUMB_PREFIX}{image_id_aerial_from}"
                with urllib.request.urlopen(url) as url_response:
                    img_data = url_response.read()
                    verdict_to_aerial = random.randint(0, 2)
                    #
                    # handle_processed_image(collections, "exterior", image_id_aerial_from,
                    #                        "exterior" if verdict_to_aerial == 1 else "interior")

                    if verdict_to_aerial in [0, 1]:
                        aerial["aerial_wrong"] = aerial["aerial_wrong"] + 1
                    else:
                        aerial["aerial_correct"] = aerial["aerial_correct"] + 1
                    result.append(
                        ProcessingImage(user_id, image_id_aerial_from, img_data,
                                        verdict_view_point_elevation=verdict_to_aerial))
            except Exception as e:
                print(e)
                print("EXCEPTION")
            counter += 1

        print(aerial)

        counter = 0

        while counter != 500:
            image_id_raised_from = random.randint(1020, 2038)
            user_id = random.randint(1, 100)
            try:
                # Ensure only 1 feedback per image from a user
                if [user_id, image_id_raised_from] in uniqueness_check:
                    print("RAISED: Uniqueness break")
                    continue
                else:
                    uniqueness_check.append([user_id, image_id_raised_from])

                url = f"{THUMB_URL}{THUMB_PREFIX}{image_id_raised_from}"
                with urllib.request.urlopen(url) as url_response:
                    img_data = url_response.read()
                    verdict_to_raised = random.randint(0, 2)
                    #
                    # handle_processed_image(collections, "exterior", image_id_aerial_from,
                    #                        "exterior" if verdict_to_aerial == 1 else "interior")

                    if verdict_to_raised in [0, 2]:
                        raised["raised_wrong"] = raised["raised_wrong"] + 1
                    else:
                        raised["raised_correct"] = raised["raised_correct"] + 1
                    result.append(
                        ProcessingImage(user_id, image_id_raised_from, img_data,
                                        verdict_view_point_elevation=verdict_to_raised))
            except Exception as e:
                print(e)
                print("EXCEPTION")
            counter += 1

        print(raised)

        counter = 0

        while counter != 500:
            image_id_ground_from = random.randint(1, 1019)
            user_id = random.randint(1, 100)
            try:
                # Ensure only 1 feedback per image from a user
                if [user_id, image_id_ground_from] in uniqueness_check:
                    print("GROUND: Uniqueness break")
                    continue
                else:
                    uniqueness_check.append([user_id, image_id_ground_from])

                url = f"{THUMB_URL}{THUMB_PREFIX}{image_id_ground_from}"
                with urllib.request.urlopen(url) as url_response:
                    img_data = url_response.read()
                    verdict_to_ground = random.randint(0, 2)
                    #
                    # handle_processed_image(collections, "exterior", image_id_aerial_from,
                    #                        "exterior" if verdict_to_aerial == 1 else "interior")

                    if verdict_to_ground in [1, 2]:
                        ground["ground_wrong"] = ground["ground_wrong"] + 1
                    else:
                        ground["ground_correct"] = ground["ground_correct"] + 1
                    result.append(
                        ProcessingImage(user_id, image_id_ground_from, img_data,
                                        verdict_view_point_elevation=verdict_to_ground))
            except Exception as e:
                print(e)
                print("EXCEPTION")
            counter += 1

        print(ground)

        return result, collections

