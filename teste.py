import cv2
import os

# Directory containing the images
input_folder = 'ImagensReal'
output_folder = 'ResizedImages'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all files in the input directory
files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# Initialize counter for output filenames
counter = 0

# Loop through all files in the input directory
for file in files:
    # Construct the full file path
    file_path = os.path.join(input_folder, file)
    
    # Read the image
    img = cv2.imread(file_path)
    
    # Check if the image was successfully loaded
    if img is not None:
        # Resize the image to 1920x1080
        resized_img = cv2.resize(img, (1920, 1080))
        
        # Construct the output file path
        output_file_path = os.path.join(output_folder, f'i{counter}.png')
        
        # Save the resized image
        cv2.imwrite(output_file_path, resized_img)
        
        # Increment the counter
        counter += 1
    else:
        print(f"Failed to load image: {file_path}")

print("Image resizing completed.")
