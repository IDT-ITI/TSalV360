
import numpy as np

resolution = (320,640)


def haversine(coord1, coord2):
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c
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
def fill_loss_frames(lists_of_results1, lists_frames1, num_frames):
    lists_of_results = []
    lists_frames = []
    print("len",len(lists_of_results1))
    print("len2",len(lists_frames1))
    print("num frames",num_frames)
    #lists_of_results1 = [sublist for sublist in lists_of_results1 if len(sublist) >= int(90 / sampler)]
    #lists_frames1 = [sublist for sublist in lists_frames1 if len(sublist) >= int(90 / sampler)]

    for i in range(len(lists_of_results1)):
        new_frames = []
        new_results = []
        for j in range(len(lists_frames1[i]) - 1):
            dif = int(lists_frames1[i][j + 1]) - int(lists_frames1[i][j])

            # Add the current frame and result (tuple)
            new_frames.append(int(lists_frames1[i][j]))
            new_results.append(lists_of_results1[i][j])

            # Check if the difference is less than or equal to num_frames
            if dif <= num_frames:

                start_result = lists_of_results1[i][j]
                end_result = lists_of_results1[i][j + 1]


                # Generate intermediate frames and interpolate each element in the tuple

                for step in range(1, dif):
                    if (abs(start_result[0] - end_result[0]) < resolution[1] / 2):
                        x = start_result[0] + (end_result[0] - start_result[0]) * (step / (dif - 1))
                        y = start_result[1] + (end_result[1] - start_result[1]) * (step / (dif - 1))

                    else:
                        if max(start_result[0], end_result[0]) == start_result[0]:
                            if end_result[0] > resolution[1]:
                                end_result[0] = resolution[1] - end_result[0]
                                x = start_result[0] + (end_result[0] - start_result[0]) * (step / (dif - 1))
                                y = start_result[1] + (end_result[1] - start_result[1]) * (step / (dif - 1))

                            else:
                                x = start_result[0] + (end_result[0] + resolution[1] - start_result[0]) * (
                                            step / (dif - 1))
                                y = start_result[1] + (end_result[1] - start_result[1]) * (step / (dif - 1))

                        else:
                            if start_result[0] < 0:
                                start_result[0] = resolution[1] + start_result[0]
                                x = start_result[0] + (end_result[0] - start_result[0]) * (step / (dif - 1))
                                y = start_result[1] + (end_result[1] - start_result[1]) * (step / (dif - 1))

                            else:
                                x = start_result[0] + (end_result[0] - resolution[1] - start_result[0]) * (
                                        step / (dif - 1))
                                y = start_result[1] + (end_result[1] - start_result[1]) * (step / (dif - 1))


                    if x > resolution[1]:
                        x = x - resolution[1]
                    if x < 0:
                        x = resolution[1] + x

                    new_results.append((x, y))
                    cc = int(lists_frames1[i][j]) + step
                    new_frames.append(cc)

            else:
                lists_of_results.append(new_results)
                lists_frames.append(new_frames)

                new_frames = []
                new_results = []

        lists_of_results.append(new_results)
        lists_frames.append(new_frames)


    final_salient_regions = [sublist for sublist in lists_of_results if len(sublist) >= 20]
    final_frames = [sublist for sublist in lists_frames if len(sublist) >= 20]
    # final_frames = [lst[:-10] for lst in final_frames]
    # final_salient_regions = [lst[:-10] for lst in final_salient_regions]
    print("length of shots",len(final_frames))
    combined = list(zip(final_frames, final_salient_regions))
    combined.sort(key=lambda x: x[0])


    # separate back into two lists
    final_frames, final_salient_regions = zip(*combined)

    return final_salient_regions, final_frames

def group_salient_regions(input_list, frame_number, fill_loss, harvs_distance, resolution):
    # Initialize a dictionary to store the groups based on the x-axis value (first element of each tuple)
    groups_dict = {}
    groups_frame = {}
    groups_centers = {}
    print("intput list",len(input_list))
    # Define image dimensions
    height, width = resolution # Replace with actual frame size
    #print("height", height)
    lonlat_centers = [
        [pixel_to_latlon(center[0],center[1], width, height) for center in frame_centers]
        for frame_centers in input_list
    ]

    # Iterate through the input list and group elements based on x-axis proximity
    for i, sublist in enumerate(lonlat_centers):

        for j, item in enumerate(sublist):
            added_to_group = False
            lower_distance = 10000
            group = 0

            for group_key in groups_dict.keys():

                last_group_items = groups_dict[group_key][-1:]

                for group_item in last_group_items:
                    a = haversine(item,group_item)

                    if a < harvs_distance:
                        lower_distance = a
                        group = group_key
                        break

                if lower_distance != 10000:
                    break

            if lower_distance != 10000 and lower_distance < harvs_distance:
                if frame_number[i] not in groups_frame[group]:
                    groups_dict[group].append(item)
                    groups_centers[group].append(input_list[i][j])
                    groups_frame[group].append(frame_number[i])
                    added_to_group = True

            if not added_to_group:
                close = False
                for group_key in groups_dict.keys():
                    group_item = groups_dict[group_key][-1:]
                    last_frame = groups_frame[group_key][-1:][0]
                    current_frame = int(frame_number[i])
                    #print("current_frame",current_frame)
                    #print("last frame",last_frame)
                    '''print("group_item",group_item[0])
                    print("item",item)'''
                    dis = haversine(item,group_item[0])

                    if dis<0.25 and current_frame-last_frame<8:


                        close = True
                        break
                if close == False:
                    # print("no close, so new item",item)

                    groups_dict[tuple(item)] = [item]
                    groups_frame[tuple(item)] = [frame_number[i]]
                    groups_centers[tuple(item)] = [input_list[i][j]]

            # Hold together the salient regions, the frames, the scores a
            groups_dict = dict(sorted(groups_dict.items(), key=lambda item: len(item[1]), reverse=True))
            groups_frame = dict(sorted(groups_frame.items(), key=lambda item: len(item[1]), reverse=True))
            groups_centers = dict(sorted(groups_centers.items(), key=lambda item: len(item[1]), reverse=True))

    # Convert the dictionary values to the final lists

    results_frames = list(groups_frame.values())
    result_lists = list(groups_centers.values())

    list_of_2d_volumes, list_of_2d_frames = fill_loss_frames(result_lists, results_frames, fill_loss)

    # print("results_frames", list_of_2d_frames)

    return list_of_2d_volumes, list_of_2d_frames