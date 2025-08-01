import pickle
import argparse
import os

import numpy as np

def latlon_to_equirectangular(lon, lat, mapwidth, mapheight):
    # Convert longitude to x
    x = (lon + 180) / 360
    # Convert latitude to y
    y = (90 - lat) / 180
    return x, y

from scipy import ndimage
import cv2

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

def convert_fix_to_sal(data, output_sal_paths):

    for item in data.keys():
        #print("length of data[item].keys()",len(data[item].keys()))
        sal_path = os.path.join(output_sal_paths,item)
        if not os.path.exists(sal_path):
            os.mkdir(sal_path)
        print(f"Processing video {item}")
        for l,it in enumerate(data[item].keys()):

            frame_data = data[item][it]
            fixations = []

            for i in range(len(frame_data)):

                fixations.append([frame_data[i][0], 1-frame_data[i][1]])

            saliency_map = salmap_from_norm_coords(np.array(fixations), sigma_deg * 960 / 360.0, (480,960))
            sal_map = os.path.join(sal_path,f"{it}.png")
            cv2.imwrite(sal_map,(saliency_map * 255).astype(np.uint8))


sigma_deg = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts fixation maps to saliency maps for sports-360 dataset.')
    parser.add_argument('--pkl_file', type=str, required=True, help='Path to pkl file, vinfo.pkl')
    parser.add_argument('--output_sal_path', type=str, required=True, help='Path to save the generated saliency maps')

    args = parser.parse_args()

    pkl_file = args.pkl_file
    with open(pkl_file, "rb") as file:  # Open in read-binary mode
        data = pickle.load(file)  # Load the data
    sigma_deg = 5
    print(type(data))  # Check the data type
    print(data.keys())  # See available keys

    output_sal_path = args.output_sal_path

    if not os.path.exists(output_sal_path):
        os.makedirs(output_sal_path)

    convert_fix_to_sal(data,output_sal_path)

