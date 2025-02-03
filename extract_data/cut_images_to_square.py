#cut images to square

from PIL import Image
import os

def crop_to_square(image_path, output_path, size):
    """
    Crop an image to a square of the specified size and save it.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
        size (int): Desired square size (width and height in pixels).
    """
    with Image.open(image_path) as img:
        # Calculate the center crop box
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim + 200
        
        # Crop to the largest centered square
        img_cropped = img.crop((left, top, right, bottom))
        
        # Resize to the desired square size
        img_resized = img_cropped.resize((size, size))
        
        # Save the cropped image
        img_resized.save(output_path)
        print(f"Saved cropped image to {output_path}")

# Example Usage
for image in os.listdir("output_frames"):
    crop_to_square("output_frames/" + image, "cropped_frames/" + image, 700)

