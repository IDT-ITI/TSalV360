
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
import numpy as np
import cv2
import hdbscan
def equirectangular_to_spherical(width, height):
    """ Convert pixel coordinates to spherical coordinates """
    lon = np.linspace(-np.pi, np.pi, width)
    lat = np.linspace(-np.pi / 2, np.pi / 2, height)
    return np.meshgrid(lon, lat)

def apply_fov_mask(saliency_map, fov, center_lon, center_lat, smooth_edge=20):
    """ Apply a smooth mask to the saliency map based on FOV """
    height, width = saliency_map.shape
    lon, lat = equirectangular_to_spherical(width, height)

    # Compute angular distance from center
    delta_lon = lon - np.radians(center_lon)
    #delta_lat = lat - np.radians(center_lat)
    angular_distance = np.arccos(
        np.sin(np.radians(center_lat)) * np.sin(lat) +
        np.cos(np.radians(center_lat)) * np.cos(lat) * np.cos(delta_lon)
    )

    # Define mask with smooth transition
    #mask = angular_distance > np.radians(fov / 2)
    smooth_mask = np.clip((np.radians(fov / 2 + smooth_edge) - angular_distance) / np.radians(smooth_edge), 0, 1)

    # Apply smooth transition instead of a hard cut-off
    saliency_map = saliency_map * smooth_mask
    return saliency_map.astype(np.uint8)
def sc_find_center_of_mass(sal_map, km=False, factor=2.0, bias=1.0, verbose=False):
    # if not kmeans is selected just return position of max val
    if not km:
        max_val = np.amax(sal_map)
        #print("max_val",max_val)
        if max_val > 0:
            [y, x] = np.unravel_index(sal_map.argmax(), sal_map.shape)
        else:
            x = None
            y = None

        return x, y
    initH = sal_map.shape[0]
    initW = sal_map.shape[1]
    sal_map = cv2.resize(sal_map, None, fx=1.0 / factor, fy=1.0 / factor, interpolation=cv2.INTER_NEAREST)

    # find max val and its indicies as the initial cluster center
    max_val = np.amax(sal_map)
    [max_row, max_col] = np.unravel_index(sal_map.argmax(), sal_map.shape)

    # init & gather points
    coo = coo_matrix(sal_map).tocoo()
    X = np.vstack((coo.row, coo.col, coo.data)).transpose().astype(float)
    max_dim = max([initH / factor, initW / factor])

    # cluster in 1 group to get mass mean
    if X.shape[0] > 0:
        X[:, 2] = (X[:, 2] / np.amax(X[:, 2])) * max_dim * bias
        X = X.astype(np.uint8)
        clusterer = KMeans(n_clusters=1, random_state=0,
                           init=np.array([[max_row, max_col, max_val]]),
                           n_init=1,
                           max_iter=5).fit(X)
        # scale back
        x = clusterer.cluster_centers_[0][1] * factor
        y = clusterer.cluster_centers_[0][0] * factor
        del clusterer
    else:
        return None, None

    return x, y
def get_correct_lonlat_bounds(cluster_points, lonlat_points):
    """
        cluster_points (numpy.ndarray): Pixel coordinates (x, y) of clusters.
        lonlat_points (numpy.ndarray): Corresponding (longitude, latitude) values.

    Returns:
        (x_min, y_min, x_max, y_max), (lon_min, lat_min, lon_max, lat_max)
    """
    # Get min/max pixel coordinates
    x_min, y_min = cluster_points.min(axis=0)
    x_max, y_max = cluster_points.max(axis=0)

    # Find indices where these values occur in cluster_points
    min_dist = np.linalg.norm(cluster_points - np.array([x_min, y_min]), axis=1)
    max_dist = np.linalg.norm(cluster_points - np.array([x_max, y_max]), axis=1)

    min_idx = np.argmin(min_dist)
    max_idx = np.argmin(max_dist)

    lon_min, lat_min = lonlat_points[min_idx]
    lon_max, lat_max = lonlat_points[max_idx]

    return (x_min, y_min, x_max, y_max), (lon_min, lat_min, lon_max, lat_max)
def pixel_to_latlon(x, y, width, height):
    """
    Convert pixel coordinates (x, y) from an ERP image to (longitude, latitude).

    Parameters:
    - x, y: Pixel coordinates
    - width, height: Dimensions of the ERP image

    Returns:
    - lon, lat: Longitude and Latitude coordinates
    """
    lon = (x / width) * 360.0 - 180.0  # Scale x to [-180, 180] degrees
    lat = (y / height) * 180.0 -90 # Scale y to [90, -90] degrees
    return lon, lat
def extract_salient_pixels(saliency_map, threshold=0.5):
    """
    Parameters:
        saliency_map: Grayscale saliency map (NumPy array).
        threshold: Saliency threshold (0-1).

    Returns:
        salient_points: List of (x, y) pixel coordinates.
    """
    y_indices, x_indices = np.where(saliency_map > threshold)
    salient_points = np.column_stack((x_indices, y_indices))
    return salient_points

def extract_salient_areas(saliency_map3,output_path, intensity_value, resolution,nu):
    saliency_map = cv2.imread(saliency_map3, cv2.IMREAD_GRAYSCALE)
    global max_width ,min_width
    res_sal = (saliency_map.shape[0], saliency_map.shape[1])
    sal_map = cv2.resize(saliency_map, (resolution[1],resolution[0])).astype(np.float32) / 255.0
    saliency_map_filtered = sal_map.copy()


    #print(resolution)
    saliency_map_filtered[sal_map <= intensity_value] = 0
    pixel_coords = extract_salient_pixels(sal_map, intensity_value)
    height, width = sal_map.shape
    # Convert (x, y) to (longitude, latitude)
    lonlat_coords = np.array([pixel_to_latlon(x, y, width, height) for x, y in pixel_coords])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=3, metric='haversine',cluster_selection_epsilon=0.01, approx_min_span_tree=True,  # True
                                     gen_min_span_tree=False,  # False
                                     cluster_selection_method='eom',  # 'eom', 'leaf'
                                     core_dist_n_jobs=4,  # 4
                                     allow_single_cluster=True)
    labels = clusterer.fit_predict(np.radians(lonlat_coords))


    unique_labels = set(labels)
    # print("unique_labels",unique_labels)
    boxes = []
    latlon_boxes = []
    for label in unique_labels:
        # print("haha")
        if label == -1:
            continue  # Skip noise points
        flag = False
        #print("label",label)
        cluster_points = pixel_coords[labels == label]

        lonlat_points = lonlat_coords[labels == label]

        #print("first try")
        for r, item in enumerate(cluster_points):
            if r > 0:
                if abs(cluster_points[r - 1][0] - cluster_points[r][0]) > int(width)/2:

                    flag = True
                    break
        #print("flag", flag)
        if flag == True:
            for r, item in enumerate(cluster_points):
                if item[0] > int(width)/2:

                    cluster_points[r][0] = item[0] - width

        #print(cluster_points)
        (pixel_bounds, lonlat_bounds) = get_correct_lonlat_bounds(cluster_points,lonlat_points)

        x_min, y_min, x_max, y_max = pixel_bounds
        x_min1, y_min1, x_max1, y_max1 = lonlat_bounds

        if x_max - x_min > 16 or y_max - y_min > 16:

            boxes.append((x_min, x_max, y_min, y_max))
            latlon_boxes.append((x_min1,x_max1,y_min1,y_max1))

    centers = []
    sal_scores = []
    for l,item in enumerate(latlon_boxes):
        x_min, x_max, y_min, y_max = item

        ll = x_min
        x_min = x_max
        x_max = ll

        if (abs(x_min)+abs(x_max)>280 and ((x_min>0 and x_max<0) or (x_min<0 and x_max>0))):
            if x_min>0:
                x_min_diff = 180-x_min
                x_max_diff = 180+x_max
                x_diff = (x_min_diff+x_max_diff)/2
                if x_min_diff>x_max_diff:
                    lon = x_min + x_diff
                else:
                    lon = x_max - x_diff
            else:
                x_min_diff = 180 + x_min
                x_max_diff = 180 - x_max
                x_diff = (x_min_diff + x_max_diff) / 2
                if x_min_diff > x_max_diff:
                    lon = x_min - x_diff

                else:
                    lon = x_max + x_diff

            #lon = float((abs(x_min1) + abs(x_max1)) / 2)
        else:
            lon = float((x_min + x_max) / 2)

        lat = float((y_min + y_max) / 2)


        masked_saliency = apply_fov_mask(sal_map*255, 30, lon, lat, smooth_edge=0)

        dx, dy = sc_find_center_of_mass(masked_saliency)
        if dx!=None or dy!=None:
            saliency_score = np.sum(masked_saliency/255)

            sal_scores.append(saliency_score)
            centers.append([dx,dy])


            #print("centers", centers)

    #centers, widths = zip(*sorted(zip(centers, widths), key=lambda x: x[0]))
    sorted_pairs = sorted(zip(sal_scores, centers), key=lambda x: x[0], reverse=True)

    # Unzip the sorted pairs
    sal_scores, centers= zip(*sorted_pairs)

    # Convert back to lists (since zip returns tuples)
    centers = list(centers)



    return centers