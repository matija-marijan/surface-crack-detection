import os
from PIL import Image

def load_convert_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img.save(os.path.join(output_folder, filename))  # Save to output folder

def load_convert_resize_and_save_images(input_folder, output_folder, size=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize(size)  # Resize the image to 64x64
            img.save(os.path.join(output_folder, filename))  # Save to output folder

# Define the paths to the input and output folders
input_neg_path = 'Dataset/Negative'
input_pos_path = 'Dataset/Positive'
output_neg_path = 'Dataset/Gray/Negative'
output_pos_path = 'Dataset/Gray/Positive'
resize_neg_path = 'Dataset/Resized/Negative'
resize_pos_path = 'Dataset/Resized/Positive'

# Load, convert, and save the images
# load_convert_and_save_images(input_neg_path, output_neg_path)
# load_convert_and_save_images(input_pos_path, output_pos_path)
# print("Images have been converted to grayscale and saved.")

# Load, convert, resize, and save the images
load_convert_resize_and_save_images(input_neg_path, resize_neg_path)
load_convert_resize_and_save_images(input_pos_path, resize_pos_path)
print("Images have been converted to grayscale, resized, and saved.")