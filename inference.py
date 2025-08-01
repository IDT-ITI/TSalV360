import os
import torch
from torch.utils.data import DataLoader
from utils.setup import get_model_test
from dotmap import DotMap
import yaml

import torch
import argparse

import numpy as np


from PIL import Image
import cv2
from torchvision import transforms
def transform_img(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ColorJitter(0.1, 0.1, 0.05, 0.1),  # Uncomment if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform(frame)

def test(video_path,text_input,target_fps, model, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(original_fps / target_fps)
    if frame_skip < 1:
        frame_skip = 1

    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_rgb = transform_img(frame_rgb)

            frames.append(frame_rgb)
        if len(frames)==8:
            clip = torch.stack(frames)

            # print("frames paths",frames_paths)

            pred_erp = model(clip.unsqueeze(0).cuda(), text_input)

            sal_map = pred_erp[0, :, :, :].squeeze(1)

            saliency_map = sal_map.squeeze(0)
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            saliency_map_np = saliency_map.detach().cpu().numpy()
            saliency_image = Image.fromarray((saliency_map_np * 255).astype(np.uint8))  # Convert to [0, 255]
            saliency_image.save(os.path.join(output_path, f'sal_map_{frame_index:04d}.png'))

            frames = []

        frame_index += 1
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a 360-degree video, given a text prompt.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--text_input', type=str, required=True, help='Text description to guided the models saliency')
    parser.add_argument('--model_path', type=str, required=False, help='Path to trained model file')
    parser.add_argument('--output_path', type=str, required=True, help='Path for the generated saliency maps to be saved')
    args = parser.parse_args()

    gpu = "cuda:0"
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config_fn = 'configs/inference.yml'

    config = DotMap(yaml.safe_load(open(config_fn, 'r')))

    path_to_video = args.video_path
    text_input = args.text_input
    output_path = args.output_path

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = ""

    model = get_model_test(config, model_path).to(DEVICE)


    with torch.no_grad():

        test(path_to_video,text_input, 16,model, output_path)