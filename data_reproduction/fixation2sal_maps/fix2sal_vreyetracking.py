import os

import numpy as np
import argparse

from PIL import Image

def load_fixation_map(image_path, threshold=200):
    """
    Loads a fixation map from a PNG image and extracts fixation points.

    Returns:
        fixations (list of [lon, lat])
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img).astype(np.float32)

    # Normalize safely
    range_val = img_array.max() - img_array.min()
    if range_val > 0:
        img_array = (img_array - img_array.min()) / range_val * 255
    else:
        img_array = np.zeros_like(img_array)

    height, width = img_array.shape
    y_coords, x_coords = np.where(img_array > threshold)

    lon = (x_coords / width) * 360.0 - 180.0
    lat = 90.0 - (y_coords / height) * 180.0
    fixations = list(zip(lon, lat))
    return fixations
from scipy import ndimage
def salmap_from_norm_coords(norm_coords, sigma, height_width):
    '''
    Base function to compute general saliency maps, given the normalized (from 0 to 1)
    fixation coordinates, the sigma of the gaussian blur, and the height and
    width of the saliency map in pixels.
    '''
    temp = norm_coords
    temp[:, 1] = 1 - norm_coords[:, 1]

    img_coords = np.mod(np.round(temp * np.array((height_width[1], height_width[0]))), np.array((height_width[1], height_width[0]))-1.0).astype(int)

    gaze_counts = np.zeros((height_width[0], height_width[1]))
    for coord in img_coords:
        gaze_counts[coord[1], coord[0]] += 1.0

    gaze_counts[0, 0] = 0.0

    sigma_y = sigma
    salmap = ndimage.filters.gaussian_filter1d(gaze_counts, sigma=sigma_y, mode='wrap', axis=0)

    # In x-direction, we scale the radius of the gaussian kernel the closer we get to the pole
    for row in range(salmap.shape[0]):
        angle = (row/float(salmap.shape[0]) - 0.5) * np.pi
        sigma_x = sigma_y / (np.cos(angle) + 1e-3)
        salmap[row,:] = ndimage.filters.gaussian_filter1d(salmap[row,:], sigma=sigma_x, mode='wrap')

    # normalize
    salmap /= np.max(salmap)
    return salmap
import cv2
map_width, map_height = 2048, 1024

def convert_fix_to_sal(fix_paths,output_sal_paths):
    vids = os.listdir(fix_paths)
    for vid in vids:
        print(f"Processing video {vid}")
        if not os.path.exists(os.path.join(output_sal_paths,vid)):
            os.makedirs(os.path.join(output_sal_paths,vid))
        fix_path = os.path.join(fix_paths)
        fx_list = os.listdir(fix_path)
        for i, fix_file in enumerate(fx_list):
            fixation_map_path = os.path.join(fix_path, fix_file)


            fixations = load_fixation_map(fixation_map_path)

            if len(fixations) == 0:
                print("No fixations found.")
                continue

            # Convert lon/lat to normalized coordinates
            fix_np = np.array(fixations)
            x_norm = (fix_np[:, 0] + 180.0) / 360.0
            y_norm = (fix_np[:, 1]-90) / 180.0
            norm_coords = np.stack([x_norm, y_norm], axis=1)

            saliency_map = salmap_from_norm_coords(norm_coords, 5 * 2048 / 360.0, [map_height, map_width])

            save_path = os.path.join(output_sal_paths, fix_file)
            resized_map = cv2.resize((saliency_map * 255).astype(np.uint8), (960, 480))
            cv2.imwrite(save_path, resized_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts fixation maps to saliency maps for vre-eyetracking dataset.')
    parser.add_argument('--fix_paths', type=str, required=True, help='Path to fixation maps')
    parser.add_argument('--output_sal_path', type=str, required=True, help='Path to save the generated saliency maps')

    args = parser.parse_args()



    fix_paths = args.fix_paths
    output_sal_path = args.output_sal_path

    if not os.path.exists(output_sal_path):
        os.makedirs(output_sal_path)

    convert_fix_to_sal(fix_paths,output_sal_path)
