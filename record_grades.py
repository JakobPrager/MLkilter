from pytesseract import image_to_string
from PIL import Image
import re
import os

# Function to extract specific text
def extract_text_from_image(image_path, regex_pattern):
    image = Image.open(image_path)
    text = image_to_string(image)
    matches = re.findall(regex_pattern, text)
    #if matches has a t in it make it a plus
    for i in range(len(matches)):
        if 't' in matches[i]:
            matches[i] = matches[i].replace('t', '+')
    if len(matches) == 0:
        return None
    print(matches[0])
    return matches[0]

# Directory containing screenshots
directory = "output_frames"

# Define regex for specific text snippet
regex_pattern = r"[4-8][a-c][+t]?"


# Iterate through all screenshots
results = {}
for filename in os.listdir(directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        file_path = os.path.join(directory, filename)
        snippets = extract_text_from_image(file_path, regex_pattern)
        results[filename] = snippets

# Save results
#for image, matches in results.items():
#    print(f"{image}: {matches}")

#save the results.items in a csv file
import csv  
with open('results.csv', 'w') as f:
    for matches in results.items():
        f.write("%s,%s\n"%(matches))


