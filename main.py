from faster_dataloader import TextClipsLoader
import torch
from torch.utils.data import DataLoader
from utils.setup import get_model, set_to_eval, get_model_2
from dataset.videoclip_dataset import VideoRead
from utils.metrics import inner_worker
from dotmap import DotMap
import yaml
import os
from utils.setup import get_model, get_criterion, get_optimizer_and_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image

import numpy as np

import time
#set_to_eval(pretrained_model)
from torchsummary import summary
#pretrained_model = pretrained_model.module.to(DEVICE)


'''for name in model.state_dict().keys():
    print(name)'''

from sklearn.model_selection import KFold
import numpy as np


def prepare_k_fold_datasets(frames_path, sal_maps_path, text_path, all_video_names,
                            load_gt, frames_per_data, n_splits=5, random_state=42):
    """
    Prepares K pairs of (train, val) datasets for cross-validation

    Args:
        n_splits: Number of folds (default 5)
        random_state: For reproducible splits
    Returns:
        List of tuples: [(train_dataset1, val_dataset1), ...]
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    video_names_array = np.array(all_video_names)

    fold_datasets = []

    for train_idx, val_idx in kf.split(video_names_array):
        train_videos = video_names_array[train_idx]
        val_videos = video_names_array[val_idx]
        print("train videos",train_videos)
        print("val videos",val_videos)
        train_dataset = TextClipsLoader(
            frames_path=frames_path,
            sal_maps_path=sal_maps_path,
            text_path=text_path,
            video_names=train_videos,
            load_gt=load_gt,
            frames_per_data=frames_per_data
        )

        val_dataset = TextClipsLoader(
            frames_path=frames_path,
            sal_maps_path=sal_maps_path,
            text_path=text_path,
            video_names=val_videos,
            load_gt=load_gt,
            frames_per_data=frames_per_data
        )

        fold_datasets.append((train_dataset, val_dataset))

    return fold_datasets

def save_checkpoint(model, optimizer, scheduler,epoch,kfold):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    out_path = r"D:\Program Files\IoanProjects\testing"
    os.makedirs(out_path, exist_ok=True)
    out_path = fr"{out_path}\Epoch_{epoch}_{kfold}.pt"
    torch.save(checkpoint, out_path)
    print('Checkpoint saved at', out_path)


def train(model,fold_datasets, output_path, load_gt,epochs):
    #print("fold datastets",fold_datasets)
    if load_gt == True:
        for fold, (train_dataset, val_dataset) in enumerate(fold_datasets):
            print(f"\n=== Training Fold {fold + 1}/{n_splits} ===")

            # Create dataloaders for this fold
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=8,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,  # No need to shuffle validation
                num_workers=8,
                pin_memory=True
            )
            for epoch in range(epochs):
                start_time = time.time()
                model.train()
                clip_counter = 0
                avg_loss_train = 0.
                for i, video in enumerate(train_loader):
                    #print("len video",len(video))
                    #for j, vid in enumerate(video):

                    clip,gt_,text, sals_paths, frames_paths = video


                    optimizer.zero_grad()
                    pred_erp,loss_vac = model(clip.cuda(),text)

                    pred_erp = pred_erp.squeeze(1)

                    clip_counter+=1
                    loss_vac = loss_vac.mean()
                    #print("loss vac",loss_vac)

                    loss_kld, loss_cc= criterion(pred_erp.cpu(), gt_.cpu(), overlap_mask)
                    #print("loss_kl, loss_cc, loss_nss vac", loss_kl, loss_cc, loss_nss)

                    loss = w_kld * loss_kld + w_cc * loss_cc
                    avg_loss_train += loss.sum().item()
                    #print("loss.sum().item()",loss.sum().item())
                    #print(loss.requires_grad)  # Should be True
                    #print(loss.grad_fn)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                scheduler.step(avg_loss_train/clip_counter)

                model.eval()
                with torch.no_grad():
                    clip_counter_eval = 0
                    avg_loss_eval = 0.
                    kld_list = []
                    cc_list = []
                    sim_list = []
                    for i, video in enumerate(val_loader):
                        clip, gt_, text, sals_paths, frames_paths = video

                        pred_erp, loss_vac = model(clip.cuda(), text)

                        pred_erp = pred_erp.squeeze(1)
                        clip_counter_eval += 1
                        loss_kld, loss_cc = criterion(pred_erp.cpu(), gt_.cpu(), overlap_mask)
                        # print("loss_kl, loss_cc, loss_nss vac", loss_kl, loss_cc, loss_nss)
                        kld, cc, sim = inner_worker(gt_.cpu(),pred_erp.cpu())
                        kld_list.append(kld)
                        cc_list.append(cc)
                        sim_list.append(sim)
                        loss = w_kld * loss_kld + w_cc * loss_cc
                        avg_loss_eval += loss.sum().item()
                        # print("loss.sum().item()",loss.sum().item())
                        # print(loss.requires_grad)  # Should be True
                        # print(loss.grad_fn)

                    print("epoch", epoch)

                    print("val Loss", avg_loss_eval / clip_counter_eval)
                    print("cc", np.mean(cc_list))
                    print("sim", np.mean(sim_list))
                    print("kld",np.mean(kld_list))

                    save_checkpoint(model,optimizer,scheduler,epoch,fold)
                    end_time = time.time()
                    print("Train Loss", avg_loss_train / clip_counter)
                    print("total epoch time", end_time - start_time)
                    #print("avg_loss_train/c1", avg_loss_train / clip_counter)


if __name__ == '__main__':
    gpu = "cuda:0"
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config_fn = 'configs/vac-eval.yml'

    config = DotMap(yaml.safe_load(open(config_fn, 'r')))
    pretrained_model = get_model(config).eval()
    set_to_eval(pretrained_model)

    pretrained_model = pretrained_model.module.to(DEVICE)

    config_fn = 'configs/vac-new-file.yml'

    config = DotMap(yaml.safe_load(open(config_fn, 'r')))
    model = get_model_2(pretrained_model,config).eval().to(DEVICE)

    config_fn = 'configs/vac-new-file-train.yml'

    config2 = DotMap(yaml.safe_load(open(config_fn, 'r')))
    criterion, loss_weights = get_criterion(config2)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config2)


    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    use_amp = config2.network.use_amp
    save_checkpoint(model, optimizer,scheduler,"initial","initial")



    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    w_kld = loss_weights['kl']
    print("wkld",w_kld)

    w_cc = loss_weights['cc']
    print("wcc",w_cc)
    w_nss = loss_weights['nss']
    overlap_mask = torch.ones([1, 480, 960])

    path_to_sal_maps = r"D:\Program Files\IoanProjects\VR\sals"
    path_to_text = r"C:\Users\ioankont\PycharmProjects\Text-Guided-360-saliency\dataset\testing\new_data\texts"
    path_to_frames = r"D:\Program Files\IoanProjects\VR\data"
    videos = os.listdir(path_to_sal_maps)
    load_gt = True
    clip_size = 8
    n_splits = 5
    fold_datasets = prepare_k_fold_datasets(
        path_to_frames, path_to_sal_maps, path_to_text, videos,
        load_gt, clip_size, n_splits
    )

    #train_data= TextClipsLoader(path_to_frames,path_to_sal_maps,path_to_text, videos, load_gt,frames_per_data=clip_size)

    #loaded_data = DataLoader(train_data, batch_size=1,num_workers=4,pin_memory=True, persistent_workers=True)
    '''loaded_data = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        num_workers=4,  # Adjust based on your CPU cores
        pin_memory=True,  # Enable for GPU training
        persistent_workers=True  # Maintains worker pools between epochs
    )'''
    path_to_save_saliency_maps = 0


    train(model, fold_datasets, output_path=path_to_save_saliency_maps, load_gt=load_gt,epochs=12)