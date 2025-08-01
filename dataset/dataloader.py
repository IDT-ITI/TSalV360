import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from multiprocessing import Pool

import random
from PIL import Image
import hashlib

def get_deterministic_seed(value):
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % (2**32)
class TextClipsLoader(Dataset):
    def __init__(self, frames_path, sal_maps_path, video_names, load_gt, frames_per_data, allow_flip, num_workers=4):
        self.frames_path = frames_path
        self.sal_maps_path = sal_maps_path
        self.load_gt = load_gt
        self.frames_per_data = frames_per_data

        self.video_names = video_names
        self.num_workers = num_workers

        # Precompute all samples

        self.samples = self._precompute_samples()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.ColorJitter(0.1, 0.1, 0.05, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.allow_flip = allow_flip

    def _precompute_samples(self):
        samples = []

        # Use parallel processing for path computation
        with Pool(self.num_workers) as p:
            results = p.map(self._process_video, self.video_names)

        for result in results:
            samples.extend(result)

        return samples

    def _process_video(self, file):
        video_samples = []
        sal_map_folder = os.path.join(self.sal_maps_path, file)
        specific_sal_maps_ = os.listdir(sal_map_folder)
        for l,item in enumerate(specific_sal_maps_):

            specific_sal_maps_paths = os.path.join(sal_map_folder,item)
            sal_maps = [f for f in os.listdir(specific_sal_maps_paths) if f.lower().endswith(('.png', '.jpg'))]

            prefix = file.split('_')[0]
            sal_paths = [
                os.path.join(specific_sal_maps_paths, fram)
                for fram in sal_maps
                if os.path.exists(os.path.join(self.frames_path, prefix, fram))
            ]

            sal_paths = sorted(sal_paths)

            text_file = os.path.join(self.sal_maps_path, file, item, "description.txt")
            with open(text_file, 'r') as f:
                text_info = f.read()
            for l in range(0, len(sal_paths), self.frames_per_data):
                flag_number = False
                prev_number = -1
                for k,path in enumerate(sal_paths[l:l + self.frames_per_data]):
                    number_of_frame = int(os.path.splitext(os.path.basename(path))[0])

                    if prev_number==-1:
                        prev_number = number_of_frame
                    elif k<8:
                        if (number_of_frame-prev_number>10): ## this is done because some events in the same folder i.e. (check TSV360_gt/003/003_0) have the same description (i.e. from 0056.png to 0129.png, while an event at 0231.png to 0280.png) To avoid creating clips with large temporal gaps

                            flag_number = True
                            break
                        else:
                            prev_number = number_of_frame

                if flag_number==False: ## so if it finds (above) that from a clip of 8 frames, there is a temporal gap between any of them (higher than 10 frames), flag it!
                    frame_group = sal_paths[l:l + self.frames_per_data]
                    if len(frame_group) == self.frames_per_data:
                        video_samples.append((frame_group, text_info))

        return video_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, text_info = self.samples[idx]
        seed = hash(idx) % (2 ** 32)  # ensure valid seed range
        random.seed(seed)
        np.random.seed(seed)
        #seed = get_deterministic_seed(frame_paths)

        # Seed RNGs


        frames = []
        gt_sal_paths = []
        frames_paths = []
        for path_to_gt in frame_paths:

            prefix = os.path.basename(os.path.dirname(path_to_gt)).split('_')[0]
            image_name = os.path.basename(path_to_gt)
            path_to_frame = os.path.join(self.frames_path, prefix, image_name)

            frames_paths.append(path_to_frame)

            def to_numpy(pic):
                mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
                img = np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
                if pic.mode == "1":
                    img = 255 * img
                img = np.transpose(img, (2, 0, 1))
                img = img.astype(np.float32)
                img = np.divide(img, 255)
                return img

            def normalize(img):
                mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((-1, 1, 1))
                std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((-1, 1, 1))
                return np.divide(np.subtract(img, mean), std)
            def preprocess(img):

                img = img.convert("RGB")
                img_np = to_numpy(img)
                img_np = normalize(img_np)
                return img_np


            img = Image.open(path_to_frame)
            frame = preprocess(img)

            '''frame = cv2.imread(path_to_frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.ColorJitter(0.1, 0.1, 0.05, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])'''
            frames.append(frame)


        gt_sal = cv2.imread(path_to_gt, cv2.IMREAD_GRAYSCALE)

        #gt_sal = cv2.resize(gt_sal, (960, 480))
        gt_sal = gt_sal.astype(np.float32) / 255.0
        gt_sal = torch.FloatTensor(gt_sal)
        gt_sal_paths.append(path_to_gt)

        gt_sal = gt_sal.unsqueeze(0)
        frames = np.stack(frames)


        shift = random.randint(0, 960)
        if self.allow_flip and shift > 0 and random.random() > 0.05:

            _, _, H, W = frames.shape
            #frames = torch.roll(frames, shifts=shift, dims=-1)
            frames = np.roll(frames, shift, axis=-1)
            gt_shift = shift // 2
            gt_sal = torch.roll(gt_sal, shifts=gt_shift, dims=-1)

        if self.allow_flip and random.random() > 0.5:

            frames = np.flip(frames, axis=-1).copy()
            gt_sal = torch.flip(gt_sal, [-1])

        return frames, gt_sal, text_info, gt_sal_paths


