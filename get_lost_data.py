from pytesseract import image_to_string
from PIL import Image
import re
import os
import pandas as pd

# Function to extract specific text
def extract_text_from_image(image_path, regex_pattern):
    image = Image.open(image_path)
    text = image_to_string(image)
    matches = re.findall(regex_pattern, text)
    #print(text)
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
        if index > 0:
            matches[0] = matches[0][:index-1] + matches[0][index:]
    if 'V' in matches[0]:
        matches[0] = matches[0].replace('V', '')
    if 'l' in matches[0]:
        matches[0] = matches[0].replace('l', '')
    if 'I' in matches[0]:
        matches[0] = matches[0].replace('I', '')
    #print(matches, text)
    
    
    print(matches)
    return matches[0]

# Directory containing screenshots
directory = "output_frames"

# Define regex for specific text snippet
regex_pattern = r"[4A5SB6C7JZ8][a-c][+t]?.*[V]"



df = pd.read_csv("results_updated.csv", header= None)

rows_with_nan = df[df.isnull().any(axis=1)]

# Extract filenames
filenames_with_nan = rows_with_nan[0].tolist()

# Print or process the filenames
print(filenames_with_nan)

#files_with_none_or_nan = list_files_with_none_or_nan(data)
#print("Files with None or NaN values:", files_with_none_or_nan)

# Iterate through all screenshots
# Iterate through the rows with NaN values and update the DataFrame
count= 0
for filename in filenames_with_nan:
    file_path = os.path.join(directory, filename)
    extracted_text = extract_text_from_image(file_path, regex_pattern)
    
    # Find the row corresponding to the filename and update missing values
    if extracted_text is not None:
        row_index = df[df[0] == filename].index[0]  # Find the row index for the filename
        df.loc[row_index, df.columns[1]] = extracted_text  # Update the second column with the extracted text
    print(count)
    count+=1
# Save the updated DataFrame back to the CSV file
df.to_csv("results_updated.csv", index=False, header=False)

print("Updated CSV saved as 'results_updated2.csv'")

