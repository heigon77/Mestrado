# Open the original file for reading
with open("ChessMovesTable.csv", "r") as original_file:
    # Read all lines from the original file
    lines = original_file.readlines()

# Create a new list to store modified lines
modified_lines = []

# Iterate through each line and remove spaces at the beginning of each element
for line in lines:
    # Split the line into individual elements
    elements = line.strip().split(",")
    # Remove spaces at the beginning of each element and join them back together
    modified_line = ",".join([element.strip() for element in elements])
    # Append the modified line to the list
    modified_lines.append(modified_line)

# Write the modified lines to a new file
with open("ChessMovesTable.csv", "w") as modified_file:
    # Write each modified line to the new file
    for line in modified_lines:
        modified_file.write(line + "\n")
