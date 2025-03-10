import pandas as pd

data = pd.read_csv("csv_data/results45.csv")
#remove the plus sign from the grades and save the data
data['grades'] = data['grades'].str.replace('+', '')
data.to_csv("csv_data/results45_noplus.csv", index=False)

data = pd.read_csv("csv_data/results45_noplus.csv")

def encode_and_save(data):
    # Create a custom sorting order for the grades
    grade_order = ['4a', '4b','4c','5a', '5b', '5c', '6a', '6b', '6c', '7a','7b','7c','8a','8b', '8c']
    
    # Create a mapping from original grades to encoded values
    grade_mapping = {grade: int(i)  for i, grade in enumerate(grade_order)}
    
    # Encode the values to labels using the custom order
    data_encoded = data.copy()
    #drop nan values
    data_encoded = data_encoded.dropna(subset=['grades'])
    data_encoded['grades'] = data_encoded['grades'].map(grade_mapping)
    #make sure the grades are integers
    data_encoded['grades'] = data_encoded['grades'].astype(int)

    # Save the encoded data to grades.csv
    data_encoded.to_csv("csv_data/grades45_noplus.csv", index=False)

encode_and_save(data)
# Check if the 'grades' column exists in the data
if 'grades' not in data.columns:
    raise KeyError("The 'grades' column is not present in the data.")


data_encoded = pd.read_csv("csv_data/grades45_noplus.csv")
# check list all nan values
nan_values = data_encoded[data_encoded['grades'].isnull()]
print(nan_values)