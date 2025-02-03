import pandas as pd

data = pd.read_csv("results_updated_no_plus.csv")

# Convert all 'S' to '5' in the first column
data.iloc[:, 1] = data.iloc[:, 1].str.replace('S', '5')

def encode_and_save(data):
    # Create a custom sorting order for the grades
    grade_order = ['4a', '4b','4c','5a', '5b', '5c', '6a', '6b', '6c', '7a','7b','7c','8a','8b', '8c']
    
    # Create a mapping from original grades to encoded values
    grade_mapping = {grade: int(i)  for i, grade in enumerate(grade_order)}
    
    # Encode the values to labels using the custom order
    data_encoded = data.copy()
    data_encoded['grades'] = data_encoded['grades'].map(grade_mapping)
    
    # Save the encoded data to grades.csv
    data_encoded.to_csv("grades_noplus.csv", index=False)
    
    # Save the original keys to a separate CSV file
    pd.DataFrame(list(grade_mapping.items()), columns=['Original', 'Encoded']).to_csv("grade_map_noplus.csv", index=False)

encode_and_save(data)
# Check if the 'grades' column exists in the data
if 'grades' not in data.columns:
    raise KeyError("The 'grades' column is not present in the data.")


data_encoded = pd.read_csv("grades_noplus.csv")
# Check if there are any NaN values in the 'grades' column of the encoded data
if data_encoded['grades'].isnull().any():
    nan_indices = data_encoded[data_encoded['grades'].isnull()].index.tolist()
    raise ValueError(f"There are NaN values in the 'grades' column of the encoded data at indices: {nan_indices}")