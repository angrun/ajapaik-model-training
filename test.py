import random

import cv2
import os

print(cv2.__version__)

# Define the directory containing your original images
input_dir = "/Users/annagrund/PycharmProjects/ajapaik-model-training/resources/scene/interior"

# Define the directory where you want to save the augmented images
output_dir = "/Users/annagrund/PycharmProjects/ajapaik-model-training/resources/scene/interior_feedback"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Function to apply image augmentation and save the augmented images
def augment_and_save(input_image_path, output_image_path):
    # Read the image
    print("HERE")
    image = cv2.imread(input_image_path)
    print(image)

    l = ["rescale", "rotate", "flip_horizontal", "brightness", "zoom"]

    augmentations = random.randint(0, 4)

    augmentations = l[augmentations]

    if "rescale" in augmentations:
        # Rescale by a random factor between 0.8 and 1.2
        scale_factor = random.uniform(0.8, 5)
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        image = scaled_image

    if "rotate" in augmentations:
        # Rotate the image by a random angle between -30 and 30 degrees
        angle = random.uniform(-30, 90)
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    if "flip_horizontal" in augmentations:
        # Flip the image horizontally with a 50% chance
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

    if "brightness" in augmentations:
        # Adjust brightness randomly
        brightness_factor = random.uniform(0.7, 5)
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    if "zoom" in augmentations:
        # Zoom in by cropping a random portion of the image
        zoom_factor = random.uniform(0.8, 1.0)
        zoomed_height = int(image.shape[0] * zoom_factor)
        zoomed_width = int(image.shape[1] * zoom_factor)
        h_start = random.randint(0, image.shape[0] - zoomed_height)
        w_start = random.randint(0, image.shape[1] - zoomed_width)
        zoomed_image = image[h_start:h_start + zoomed_height, w_start:w_start + zoomed_width]
        image = cv2.resize(zoomed_image, (image.shape[1], image.shape[0]))

    # Save the augmented image
    cv2.imwrite(output_image_path, image)


# Iterate through the original images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, "u" + filename)

        # Apply augmentation and save the augmented image
        augment_and_save(input_image_path, output_image_path)

# You can apply more transformations and augmentations as needed
