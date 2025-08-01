
import numpy as np
import cupy as cp
import cv2
import os
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
def xyz2lonlat(xyz):
    atan2 = cp.arctan2
    asin = cp.arcsin
    norm = cp.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]
    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]
    out = cp.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * cp.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (cp.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = cp.concatenate(lst, axis=-1)
    return out


class Equirectangular:
    def __init__(self, img, final_resolution):
        # converting the img to a CuPy array
        self._img = cp.array(img)
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        f = 0.5 * width * 1 / cp.tan(0.5 * FOV / 180.0 * cp.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0

        K = cp.array([[f.get(), 0, cx], [0, f.get(), cy], [0, 0, 1], ], cp.float32)

        K_inv = cp.linalg.inv(K)

        x = cp.arange(width)
        y = cp.arange(height)
        x, y = cp.meshgrid(x, y)
        z = cp.ones_like(x)
        xyz = cp.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = cp.array([0.0, 1.0, 0.0], cp.float32)
        x_axis = cp.array([1.0, 0.0, 0.0], cp.float32)
        # R1= rodrigues_rotation_vector_to_matrix(y_axis * cp.radians(THETA))
        # R2= rodrigues_rotation_vector_to_matrix((R1 @ x_axis) * cp.radians(PHI))
        R1, _ = cv2.Rodrigues(y_axis.get() * np.radians(THETA))
        R2, _ = cv2.Rodrigues((R1 @ x_axis.get()) * np.radians(PHI))
        R = cp.array(R2 @ R1)
        xyz = xyz @ R.T

        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(cp.float32)

        # Use get() to convert CuPy arrays to NumPy arrays for OpenCV functions
        persp = cv2.remap(self._img.get(), XY[..., 0].get(), XY[..., 1].get(), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        return persp

def fov_extractor(video_path, name,path, centers, centers1, selected_frames, final_resolution):

    print("len selected_frames", len(selected_frames))
    frames_positions = []
    count = 0
    for item in centers1:
        frames_positions.append((selected_frames[count], selected_frames[count + len(item) - 1]))
        count += len(item)
    width = 420
    height = 420

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    iii = 0
    fps = 25
    #print("centers")
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    cap = cv2.VideoCapture(video_path)


    for r,pos in enumerate(frames_positions):
        output_path = os.path.join(path,f"{name}_{r}.mp4")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos[0])

        for j in range(pos[1] - pos[0] + 1):
            ret, frame = cap.read()

            if not ret:
                break
            equ = Equirectangular(frame, final_resolution)  # Load equirectangular image

            longitude = (centers[iii][0] / final_resolution[1]) * 360 - 180
            latitude = (centers[iii][1] / final_resolution[0]) * 180 - 90
            #longitude = centers[r][j][0]
            #latitude = centers[r][j][1]
            #print("longitude",longitude,"lati",latitude)
            img = equ.GetPerspective(40, longitude, latitude, height,
                                     width)  # Specify parameters(FOV, theta, phi, height, width)
            # cv2.imwrite(f"fov_/out_{iii}.png", img)
            video_writer.write(img)
            iii += 1
    cap.release()
    cv2.destroyAllWindows()

    del cap

def extract_2d_videos(video_path,name,gt_path, ots,lists_of_results, lists_frames, resolution):
    list_centers = []
    cap = cv2.VideoCapture(video_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the video
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the video
    res = (video_height,video_width)

    print("NAME",name)

    list_centers1 = []
    selected_frames = []


    '''for i, subs_centers in enumerate(lists_of_results):
        subs_interpolated = interpolate_centers(subs_centers, num_frames_between_interpolations, i)
        list_centers.append(subs_interpolated)
        for j, it in enumerate(subs_centers):
            selected_frames.append(lists_frames[i][j])
        centers = []
        for l, current in enumerate(subs_interpolated):
            center_x = int(current[0] * multiplier)
            center_y = int(current[1] * multiplier)
            centers.append([int(center_x), int(center_y)])
            list_centers1.append([int(center_x), final_resolution[0] - int(center_y)])'''
    final_centers = []
    for i, subs_centers in enumerate(lists_of_results):
        #subs_interpolated = interpolate_centers(subs_centers, num_frames_between_interpolations, i)
        list_centers.append(subs_centers)
        #print("frames",lists_frames[i])
        cc1 = []
        for j, it in enumerate(subs_centers):

            selected_frames.append(lists_frames[i][j])
        centers = []
        for l, current in enumerate(subs_centers):
            center_x = int(current[0]* (video_width / resolution[1]))
            center_y = int(current[1]*(video_height/resolution[0]))
            centers.append([int(center_x), int(center_y)])
            list_centers1.append([int(center_x), video_height - int(center_y)])
            cc1.append([int(center_x), video_height - int(center_y)])
        final_centers.append(cc1)


    for rr,items in enumerate(lists_frames):
        name2 = f"SalMaps_Volume_{rr}"
        output_fovs_path2 = os.path.join(ots, name2)
        if not os.path.exists(output_fovs_path2):
            os.makedirs(output_fovs_path2)
        for l,it in enumerate(items):
            formatted2 = str(it).zfill(4)  # '0010'

            gt_ = os.path.join(gt_path, f"{formatted2}.png")
            sal_map = cv2.imread(gt_, cv2.IMREAD_GRAYSCALE)
            saliency_map2 = cv2.resize(sal_map, (960, 480))
            # print("saliency_map2",saliency_map2.shape)
            # print("l+10",l+10,cc)
            cens = lists_of_results[rr][l]
            fov = 30
            multiplier2 = 960/640
            #print(cens)
            center_lon = round(cens[0]*multiplier2)
            center_lat = round(cens[1]*multiplier2)

            longitude = (center_lon / 960) * 360 - 180
            latitude = (center_lat / 480) * 180 - 90

            masked_saliency = apply_fov_mask(saliency_map2, fov, longitude, latitude)
            img_pth = os.path.join(output_fovs_path2,f"{formatted2}.png")
            #print("img_pth",img_pth)
            cv2.imwrite(img_pth,masked_saliency)

    #fov_extractor2(video_path,name, fps, list_centers1, list_centers, selected_frames, final_resolution)
    final_resolution = (video_height,video_width)
    fov_extractor(video_path,name,ots, list_centers1, list_centers, selected_frames,final_resolution)

    return final_centers,res