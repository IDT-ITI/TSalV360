from salient_regions import extract_salient_areas
from create_subvolumes import group_salient_regions
from extract_2d_videos import extract_2d_videos
import re
import os
import cv2

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None  # Convert to int


init_path = r"D:\Program Files\IoanProjects\vreyetracking-salientcymaps"
frames_path = r"D:\Program Files\IoanProjects\VR\data"
jpg_files = [f for f in os.listdir(init_path) if f.endswith(".png")]
output_fovs_inD = r"D:\Program Files\IoanProjects\PVS-HM-test"
paths = os.listdir(frames_path)
vids = r"D:\Program Files\IoanProjects\VR\DatasetVideos"
for i ,it in enumerate(paths):

    if it=="034":
        current_saliency_folder = os.path.join(init_path ,it)
        current_frames_folder = os.path.join(frames_path ,it)
        sal_maps= os.listdir(current_saliency_folder)
        max_width = -2
        min_width = 1000

        cc_frame = []
        latlon_centers_all = []

        ll = 0
        output_fovs_path = os.path.join(output_fovs_inD, it)

        vs = os.path.join(vids ,f"{it}.mp4")
        vcap = cv2.VideoCapture(vs)
        fr = vcap.get(cv2.CAP_PROP_FPS)

        frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

        desired_fps = 16
        threshold = 0.35
        resolution = (320,640)
        fill_loss = 25
        harvs_distance =0.4
        frame_interval = round(fr / desired_fps)


        del vcap
        # print("fps of video",fr)

        if not os.path.exists(output_fovs_path):
            os.makedirs(output_fovs_path)
            print(f"Folder created: {output_fovs_path}")

        for j in range(len(sal_maps ) -10):

            curr_saliency_map = os.path.join(current_saliency_folder ,sal_maps[j])


            number_of_fr = sal_maps[j].replace(".png", "")

            new_file = sal_maps[j].replace("_gt.png", ".jpg")
            curr_frame_path = os.path.join(current_frames_folder ,f"{sal_maps[j]}")


            number = extract_number(number_of_fr)
            if number% 8==0 and number!=0 and number>20 and os.path.exists(curr_frame_path) and os.path.exists \
                        (curr_saliency_map):
                print("number" ,number)
                # print("yoyo")
                # print(j)
                ll+=1
                output_sal_map_path = os.path.join(output_fovs_path ,sal_maps[ j +10])


                latlon_centers = extract_salient_areas(curr_saliency_map ,output_sal_map_path
                                                                        ,threshold, resolution ,number)

                cc_frame.append(number)
                latlon_centers_all.append(latlon_centers)

        if ll>0:
            result_lists, frames = group_salient_regions(latlon_centers_all, cc_frame, fill_loss, harvs_distance
                                                         ,resolution=[resolution[0], resolution[1]])

            video_path = os.path.join(vids ,f"{it}.mp4")
            fin_cc, ff_res = extract_2d_videos(video_path ,it, current_saliency_folder, output_fovs_path ,result_lists, frames ,resolution=resolution)
            video_name = it
            data_file = os.path.join(output_fovs_path ,"data_info.txt")
            with open(data_file, 'w') as f:
                f.write(f"Video: {video_name}\n")
                f.write(f"Number_of_2d_volumes: {len(frames)}\n")
                f.write(f"Frames: {frames}\n")
                f.write(f"Centers: {fin_cc}\n")
                f.write(f"Centers_Resolution {resolution}\n")
                f.write(f"Video_Resolution: {ff_res}")