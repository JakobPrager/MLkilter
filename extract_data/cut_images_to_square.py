#cut images to square

from PIL import Image
import os

def crop_to_square(root_path,image_path, output_path, size):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    """
    Crop an image to a square of the specified size and save it.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
        size (int): Desired square size (width and height in pixels).
    """
    with Image.open(root_path+image_path) as img:
        # Calculate the center crop box
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2 - 76
        right = left + min_dim
        bottom = top + min_dim + 160
        
        # Crop to the largest centered square
        img_cropped = img.crop((left, top, right, bottom))
        
        # Resize to the desired square size
        #img_resized = img_cropped.resize((size, size))
        
        # Save the cropped image
        img_cropped.save(output_path+image_path)
        print(f"Saved cropped image to {output_path+image_path}")

"""# Example Usage
for image in os.listdir("40degree_routes"):
    crop_to_square("40degree_routes/", image, "rectangle40/", 700)"""

crop_to_square("rectangle40/", 'frame_0000.jpg', "test/", 700)