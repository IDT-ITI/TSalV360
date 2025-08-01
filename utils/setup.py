import os
import torch
from model.TSalV360 import TSalV360

from utils.saliency_losses import SaliencyLoss


def get_criterion(config, model=None):
    criterion = SaliencyLoss(config)
    loss_weights = config['train']['criterion']['weights']
    return criterion, loss_weights


def get_optimizer(model, config):
    
    # Optimizer
    optimizers = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    }

    optim_algorithm = config['train']['optim_algorithm']
    optim = config['train']['optim'][optim_algorithm]

    optimizer = optimizers[optim_algorithm](filter(lambda p: p.requires_grad, model.parameters()), **optim)
    return optimizer


def get_model_test(config,model_path="", erp_size=(960, 1920)):
    def load_pretrained(model, model_path, config):
        if model_path != "":
            ckpt = model_path
        else:

            ckpt = config.network.resume

        if ckpt:
            assert os.path.exists(ckpt), "Checkpoint does not exist!"
            # print('Loading state dict from:', os.path.basename(ckpt))
            pt_model = torch.load(os.path.abspath(os.path.expanduser(ckpt)), map_location='cpu')

            pt_model  = pt_model['model']
            ckpt = {k.replace("module.", ""): v for k, v in pt_model.items()}
            '''for name in ckpt.keys():
                print("haha",name)'''

            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            for param in model.parameters():
                param.requires_grad = False
            # print("Model state dict keys:")
            '''for name in model.state_dict().keys():
                print(name)'''

            if len(missing) > 0:
                print(f"Missing keys: {missing}")

            if len(unexpected) > 0:
                print(f"Unexpected keys: {unexpected}")
        else:
            print("No checkpoint provided. Training from scratch.")
        return model



    model = TSalV360(config, erp_size)
    model = load_pretrained(model, model_path,config)
    model = torch.nn.DataParallel(model).cuda()
    return model
def get_model(config):

    print("config.train.criterion.vac", config.train.criterion.vac)

    model = TSalV360(config)

    #model = load_pretrained(model, config)
    model = torch.nn.DataParallel(model).cuda()
    return model


def set_to_eval(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False

    model.requires_grad_(False)