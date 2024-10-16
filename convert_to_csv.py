import pandas as pd

# Load .data file and assign column names
columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
data = pd.read_csv('poker-hand-training-true.data', header=None, names=columns)

# Save as a CSV file
data.to_csv('poker_hand.csv', index=False)
print("CSV file created successfully!")
