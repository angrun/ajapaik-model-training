import requests
import os
import json

# Load the JSON data
with open('output_data_filtered.json', 'r') as json_file:
    data = json.load(json_file)

# Create a directory to save the images
if not os.path.exists('image_folder'):
    os.mkdir('image_folder')

# Iterate through the JSON data and download images
for item in data:
    image_url = item['image']
    image_filename = os.path.join('image_folder', f'image_{item["id"]}.jpg')

    # Download the image
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(image_filename, 'wb') as image_file:
            image_file.write(response.content)
        print(f"Downloaded and saved {image_filename}")
    else:
        print(f"Failed to download {image_filename}")

print("Images downloaded and saved to 'image_folder'.")