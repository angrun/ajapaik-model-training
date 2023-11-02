# import requests
# import os
# import json
#
# # Load the JSON data
# with open('opendata.json', 'r') as json_file:
#     data = json.load(json_file)
#
# # Create a directory to save the images
# if not os.path.exists('image_folder'):
#     os.mkdir('image_folder')
#
# # Iterate through the JSON data and download images
# for item in data['results']:
#     image_url = item['image']
#     image_filename = os.path.join('image_folder', f'image_{item["id"]}.jpg')
#
#     # Download the image
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         with open(image_filename, 'wb') as image_file:
#             image_file.write(response.content)
#         print(f"Downloaded and saved {image_filename}")
#     else:
#         print(f"Failed to download {image_filename}")
#
# print("Images downloaded and saved to 'image_folder'.")

import json
import requests

# Initialize variables
filtered_data = []
offset = 1149001 # run this when you are back from vacation
filtered_count = 0

while filtered_count < 250:
    # print(offset)
    url = f"https://opendata.ajapaik.ee/photos/?offset={offset}"
    print(url)
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Filter records where viewpoint_elevation is 1
        records = data.get('results', [])
        filtered_records = [record for record in records if record.get('viewpoint_elevation') == 2]

        if filtered_records:
            # Add filtered records to the overall list
            filtered_data.extend(filtered_records)
            filtered_count += len(filtered_records)
            print(filtered_count)

        # Increment the offset for the next request
        offset += len(records)

        if not records:
            # If there are no more records, exit the loop
            break
    else:
        print(f"Failed to fetch data from the URL. Status code: {response.status_code}")
        offset += 100
        continue

# Save the filtered JSON data to a file
output_file = "output_data_filtered.json"
with open(output_file, 'w') as json_file:
    json.dump(filtered_data, json_file, indent=4)

print(f"Total filtered data (viewpoint_elevation = 1) items: {filtered_count}")
print(f"Filtered data saved to {output_file}")
