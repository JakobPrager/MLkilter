#wreite code that counts the number of unique values in grades no plus csv  
import pandas as pd
data = pd.read_csv("grades_new.csv")
print(data['grades'].value_counts())
#order them
print(data['grades'].value_counts().sort_index())
#print only the amount
print(data['grades'].value_counts().sort_index().values)

#Write a function that parses the dataframe and takes only max 100 of each keay and values per values and puts the int anew csv called balanced
def balance_data(data):
    data = data.groupby('grades').head(100)
    data.to_csv('balanced.csv', index=False) 
balance_data(data)
#check if it worked
data = pd.read_csv("balanced.csv")
print(data['grades'].value_counts())
