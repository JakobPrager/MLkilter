#wreite code that counts the number of unique values in grades no plus csv  
import pandas as pd
data = pd.read_csv("csv_data/results45degrees.csv")
print(data['grades'].value_counts())
#order them
print(data['grades'].value_counts().sort_index())
#print only the amount
print(data['grades'].value_counts().sort_index().values)

#Write a function that parses the dataframe and takes only max 100 of each keay and values per values and puts the int anew csv called balanced
def balance_data(data):
    data = data.groupby('grades').head(100)
    data.to_csv('balanced.csv', index=False) 
#balance_data(data)
#check if it worked
#data = pd.read_csv("balanced.csv")
#print(data['grades'].value_counts())

#write code that gives me all indices of values with more than 2 characters where the third character is not a plus
"""data = pd.read_csv("csv_data/results45degrees.csv") 
for i in data.index:
    if data['grade'][i] == 'None':
    #skip the None values
        continue
    #fix if grade is a float   
    if type(data['grade'][i]) == float:
        data['grade'][i] = str(data['grade'][i])
    if data['grade'][i] == 'na+':
        data['grade'][i] = 'None'
#save
data.to_csv('csv_data/results45degrees.csv', index=False)
"""


