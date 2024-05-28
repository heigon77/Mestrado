import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('cleaned_dataset_output.csv')

# Create two DataFrames: one for black moves and one for white moves
black_moves = df[df['Color_To_Move'] == 'Black']
white_moves = df[df['Color_To_Move'] == 'White']

# Save the black moves DataFrame to a CSV file
black_moves.to_csv('black_moves.csv', index=False)

# Save the white moves DataFrame to a CSV file
white_moves.to_csv('white_moves.csv', index=False)