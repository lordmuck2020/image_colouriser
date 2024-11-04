import os
from PIL import Image
import numpy as np
from datetime import datetime
import random 

def load_and_preprocess_images(folder_path, target_size=(256, 256)):
    # Original folder path
    original_folder_path = folder_path + "Colour\\"
    numpy_folder_path = folder_path + "Numpy\\"
    numpy_colour_folder_path = numpy_folder_path + "Colour\\"
    numpy_grayscale_folder_path = numpy_folder_path + "Grayscale\\"

    # Create directories if they don't exist
    os.makedirs(numpy_folder_path, exist_ok=True)
    os.makedirs(numpy_colour_folder_path, exist_ok=True)
    os.makedirs(numpy_grayscale_folder_path, exist_ok=True)

    # List to store preprocessed images
    preprocessed_images = []

    # Iterate over all files in the directory
    for filename in os.listdir(original_folder_path):
        print(f"Processing {filename}")
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
            # Construct full file path
            file_path = os.path.join(original_folder_path, filename)
            
            # Open the image file
            with Image.open(file_path) as img:
                # new filename with timestamp to the milliseconds
                new_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # Resize image to target size
                img = img.resize(target_size)
                
                # Convert to numpy array and normalize color image
                color_img_array = np.array(img, dtype=np.float32) / 255.0

                # Save as numpy colour image
                np.save(os.path.join(numpy_colour_folder_path, "clr_" + new_filename + ".npy"), color_img_array)

                # Convert image to grayscale
                grayscale_img = img.convert("L")
                
                # Convert to numpy array and normalize grayscale image
                grayscale_img_array = np.array(grayscale_img, dtype=np.float32) / 255.0
                
                # Save grayscale image as numpy array
                np.save(os.path.join(numpy_grayscale_folder_path, "bw_" + new_filename + ".npy"), grayscale_img_array)

    return preprocessed_images

# Example usage
folder_path = 'C:\\Users\\JamesHancock\\OneDrive\\03_Programming\\04_AI\\ImageColouriser\\TestImages\\'

preprocessed_images = load_and_preprocess_images(folder_path)

