from PIL import Image, ImageEnhance
from vpr.data.datasets import Nordlands_passes
import os
import random
import os
from tqdm import tqdm

augmentation_type = "illumination"

def random_illumination(image_path, output_path):
    # Load the image from disk
    image = Image.open(image_path)

    # Generate random illumination adjustment factors
    brightness_factor = random.uniform(0.5, 1.5)   # Random value between 0.5 and 1.5
    contrast_factor = random.uniform(0.8, 1.2)     # Random value between 0.8 and 1.2
    sharpness_factor = random.uniform(0.5, 1.5)    # Random value between 0.5 and 1.5

    # Adjust illumination levels (brightness, contrast, sharpness)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)

    # Save the modified image back to disk
    image.save(output_path)

def alter_path(img_path, augmentation_type, N_paths=3):
    q = img_path.split('/')
    for i, folder in enumerate(q):
        if "Nordlands" in folder:
            break
    q[i + 2] += '_' + augmentation_type
    new_pth_folder = '/'.join(q[:-1])
    new_pth = '/'.join(q)
    if not os.path.exists(new_pth_folder):
        os.makedirs(new_pth_folder)

    paths = []
    for i in range(N_paths):
        paths.append(new_pth[:-4] + '_aug' + str(i) + '.png')
    return paths


if __name__ == "__main__":

    Q = Nordlands_passes.get_query_paths(partition="train")
    for q in tqdm(Q, desc="augmenting dataset"):
        new_paths = alter_path(q, augmentation_type, N_paths=1)
        for path in new_paths:
            random_illumination(q, path)

    # Set the paths for input and output images
    #input_image_path = "path/to/input/image.jpg"
    #output_image_path = "path/to/output/image.jpg"

    #random_illumination(input_image_path, output_image_path)

    #print("Random illumination adjustment applied and image saved successfully!")

