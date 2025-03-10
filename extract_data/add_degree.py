import pandas as pd
# read results45degree.csv and add the column degree and add 45 to each row in the column
df = pd.read_csv("csv_data/results40.csv")
df['degree'] = 40
df.to_csv("csv_data/results40.csv", index=False)

