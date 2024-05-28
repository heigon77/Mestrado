import numpy as np
import csv

# Load the numpy array from the .npy file
distance_matrix = np.load('concatenated_matrix_complete.npy')

# Define the filename for the CSV file
csv_filename = 'concatenated_matrix_complete.csv'

# Write the numpy array to a CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for row in distance_matrix:
        csv_writer.writerow(row)
