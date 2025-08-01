import cv2
import os
import argparse


target_fps = 16
erp_size = (1920, 960)  # (width, height)


def extract_frames(video_path, output_path):

    for i,vid in enumerate(os.listdir(video_path)):


        # Read the video from specified path if your videos are in other format change extention .mp4
        cam = cv2.VideoCapture(os.path.join(video_path, vid))
        fram_path = os.path.join(output_path, vid[:-4])
        if not os.path.exists(fram_path):
            os.mkdir(fram_path)
        orig_fps = cam.get(cv2.CAP_PROP_FPS)
        frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / orig_fps

        # Generate timestamps for 16 FPS
        frame_times = [i / target_fps for i in range(int(duration_sec * target_fps))]

        saved = 0
        for t in frame_times:
            cam.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cam.read()
            if not ret:
                break
            resized = cv2.resize(frame, erp_size)
            frame_number = int(round(t * orig_fps))  # frame index based on time
            if frame_number>20: #skip first 20 frames
                filename = f"{frame_number:04d}.png"
                name = os.path.join(output_path, vid[:-4],filename)
                cv2.imwrite(name, resized)

                saved += 1

        cam.release()
        print(f"Saved {saved} frames at exact 16 FPS using timestamp sampling.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame extraction from erp VR-Eyetracking videos.')
    parser.add_argument('--path_to_videos', type=str, required=True, help='Path to input video')

    args = parser.parse_args()
    path_to_videos = args.path_to_videos
    output_path = "dataset/frames"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    extract_frames(path_to_videos,output_path)

