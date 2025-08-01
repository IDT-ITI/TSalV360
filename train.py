

import torch
from torch.utils.data import DataLoader
from utils.metrics import inner_worker
from dotmap import DotMap
import yaml
import os
from dataset.dataloader import TextClipsLoader
from utils.setup import get_model, get_criterion, get_optimizer

from PIL import Image

import time
import random, numpy as np, torch
from sklearn.model_selection import KFold, train_test_split

def set_global_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set this at the start of your script
set_global_seed(42)
def init_log_file(log_path, config_dict):
    with open(log_path, 'a') as f:
        f.write("\n=== New Training Run ===\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("=== Training Log ===\n")
def log_epoch(log_path, fold, epoch, train_loss, val_loss, cc, sim, kld, kld2, epoch_time):
    with open(log_path, 'a') as f:
        line = (
            f"Training fold {fold}\t"
            f"Epoch {epoch}\t"
            f"TrainLoss: {train_loss:.4f}\t"
            f"CC: {cc:.4f}\t"
            f"SIM: {sim:.4f}\t"
            f"KLD: {kld:.4f}\t"
            f"KLD2: {kld2:.4f}\t"
            f"Time: {epoch_time:.2f}s\n"
        )
        f.write(line)

def prepare_k_fold_datasets_from_json(
    json_path,
    frames_path,
    sal_maps_path,
    load_gt,
    clip_size
):

    with open(json_path, 'r') as f:
        folds_data = json.load(f)

    fold_datasets = []

    for fold in folds_data:
        train_videos = fold["train_videos"]
        test_videos = fold["test_videos"]

        print(f"\nFold {fold['fold']}")
        print("Train videos:", train_videos)
        print("Validation videos:", test_videos)

        train_dataset = TextClipsLoader(
            frames_path=frames_path,
            sal_maps_path=sal_maps_path,
            video_names=train_videos,
            load_gt=load_gt,
            frames_per_data=clip_size,
            allow_flip=True
        )

        test_dataset = TextClipsLoader(
            frames_path=frames_path,
            sal_maps_path=sal_maps_path,
            video_names=test_videos,
            load_gt=load_gt,
            frames_per_data=clip_size,
            allow_flip=False
        )

        fold_datasets.append((train_dataset, test_dataset))

    return fold_datasets

def save_checkpoint(model, optimizer,epoch,kfold, save_model_folder):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None,
        'epoch': epoch,
    }

    out_path = f"{save_model_folder}/TSalV360_kfold_{kfold}.pt"
    torch.save(checkpoint, out_path)
    print('Model saved at', out_path)


def train(model,fold, train_dataset, test_dataset, batch_size, load_gt,epochs, log_path,save_model_folder):

    if load_gt == True:
        # Create dataloaders for this fold
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=5,
            pin_memory=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation
            num_workers=5,
            pin_memory=False
        )

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            clip_counter = 0
            avg_loss_train = 0.
            for i, video in enumerate(train_loader):

                clip,gt_,text, sals_paths = video
                #print("sals path",sals_paths)
                optimizer.zero_grad()
                pred_erp = model(clip.cuda(),text)

                pred_erp = pred_erp.squeeze(1)
                gt_ = gt_.squeeze(1)
                clip_counter+=1
                #print(pred_erp.mean())
                #print("gt_ mean",gt_.mean())
                loss_kld, loss_cc= criterion(pred_erp.cuda(), gt_.cuda())

                loss = w_kld * loss_kld + w_cc * loss_cc
                print("loss", loss)
                #print("loss",loss)
                avg_loss_train += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 5.0)
                optimizer.step()

            print("total train loss", avg_loss_train/clip_counter)


        model.eval()
        with torch.no_grad():
            clip_counter_eval = 0
            kld_list = []
            cc_list = []
            sim_list = []
            kld2_list = []
            for i, video in enumerate(test_loader):

                clip, gt_, text, sals_paths = video

                pred_erp = model(clip.cuda(), text)
                pred_erp = pred_erp.squeeze(1)
                gt_ = gt_.squeeze(1)
                clip_counter_eval += 1

                for i in range(gt_.shape[0]):
                    kld, cc, sim, kld2 = inner_worker(pred_erp[i].cpu(),gt_[i].cpu())

                    kld_list.append(kld)
                    cc_list.append(cc)
                    sim_list.append(sim)
                    kld2_list.append(kld2)

        print("epoch", epoch)

        print("cc", np.mean(cc_list))
        print("sim", np.mean(sim_list))
        print("kld",np.mean(kld_list))
        print("kld2 ",np.mean(kld2_list))

        save_checkpoint(model,optimizer,epoch,fold,save_model_folder)
        end_time = time.time()
        print("Train Loss", avg_loss_train / clip_counter)
        print("total epoch time", end_time - start_time)
        log_epoch(log_path, fold, epoch, avg_loss_train / clip_counter,  np.mean(cc_list), np.mean(sim_list), np.mean(kld_list), np.mean(kld2_list), end_time - start_time)


if __name__ == '__main__':

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config_fn = 'configs/train.yml'
    log_path = "training_logs.txt"
    config = DotMap(yaml.safe_load(open(config_fn, 'r')))

    init_log_file(log_path, config)

    path_to_sal_maps = config['paths']['path_to_text_saliency_maps']
    path_to_erp_frames = config['paths']['path_to_erp_frames']
    save_model_folder = config['paths']['save_model_folder']


    videos = os.listdir(path_to_sal_maps)

    clip_size = config['train']['clip_size']
    epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    kfold_splits_json = config['train']['dataset_folds_file']

    fold_datasets = prepare_k_fold_datasets_from_json(kfold_splits_json,path_to_erp_frames, path_to_sal_maps,True, clip_size)

    for fold, (train_dataset, test_dataset) in enumerate(fold_datasets):
        if fold>-1:
            print(f"\n=== Training Fold {fold + 1}/{n_splits} ===")
            #model, optimizer1, scheduler1, epoch = get_model_4(config)
            model = get_model(config)
            model = model.to(DEVICE)

            for name, param in model.named_parameters():

                if 'visual_features_extractor' in name or 'clip' in name or 'extract_resnet_features' in name or "running" in name:

                    param.requires_grad = False
                else:
                    param.requires_grad = True
            criterion, loss_weights = get_criterion(config)
            optimizer = get_optimizer(model, config)


            w_kld = loss_weights['kl']
            w_cc = loss_weights['cc']

            # K-Fold training
            print("batch size",batch_size)

            train(model, fold, train_dataset,test_dataset,batch_size, load_gt=load_gt, epochs=epochs, log_path = log_path, save_model_folder=save_model_folder)
