from pytesseract import image_to_string
from PIL import Image
import re
import os

def extract_text_from_image(image_path, regex_pattern):
    image = Image.open(image_path)
    text = image_to_string(image)
    matches = re.findall(regex_pattern, text)
    #print('matches',matches,'text',text)
    if len(matches) == 0:
        return None
    #if matches has a t in it make it a plus
    for i in range(len(matches)):
        if 't' in matches[i]:
            matches[i] = matches[i].replace('t', '+')
        if 'B' in matches[i]:
            matches[i] = matches[i].replace('B', '5')
        if 'S' in matches[i]:
            matches[i] = matches[i].replace('S', '5')
        if 'J' in matches[i]:
            matches[i] = matches[i].replace('J', '7')
        if 'Z' in matches[i]:
            matches[i] = matches[i].replace('Z', '7')
        if 'A' in matches[i]:
            matches[i] = matches[i].replace('A', '4')
    if len(matches) > 0 and 'V' in matches[0]:
        index = matches[0].index('V')
        if index >= 3:
            matches[0] = matches[0][:index-1] + matches[0][index:]
    if 'V' in matches[0]:
        matches[0] = matches[0].replace('V', '')
    if 'l' in matches[0]:
        matches[0] = matches[0].replace('l', '')
    if 'I' in matches[0]:
        matches[0] = matches[0].replace('I', '')
    #replace the 4 with a plus
    if '4' in matches[0][2]:
        matches[0][2]=='+'
    #truncate blank spaces
    matches[0] = matches[0].replace(' ', '')



    print(matches)
    return matches[0]

# Directory containing screenshots
directory = "45degree_routes"

# Define regex for specific text snippet
regex_pattern = r"[4A5SB6C7JZ8][a-c][+t]?.*[V]"


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
with open('csv_data/results45degree.csv', 'w') as f:
    for matches in results.items():
        f.write("%s,%s\n"%(matches))


