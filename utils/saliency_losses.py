import torch
import torch.nn as nn

class SaliencyLoss(nn.Module):
    def __init__(self, config):
        super(SaliencyLoss, self).__init__()


        self.eps = torch.finfo(torch.float32).eps

    def KL_loss(self, pred, target):
        pred = pred.float()
        target = target.float()

        pred = pred / (pred.sum(dim=[-2, -1], keepdim=True) + self.eps)
        target = target / (target.sum(dim=[-2, -1], keepdim=True) + self.eps)

        loss = (target * torch.log((target + self.eps) / (pred + self.eps))).sum(dim=[-2, -1]).mean()
        return loss


    def CC_loss(self, pred, target):
        pred = pred.float()
        target = target.float()
        pred_mean = pred.mean([-2, -1], keepdim=True)
        pred_std = pred.std([-2, -1], keepdim=True)
        pred = (pred - pred_mean) / (pred_std + self.eps)

        target_mean = target.mean([-2, -1], keepdim=True)
        target_std = target.std([-2, -1], keepdim=True)
        target = (target - target_mean) / target_std
        loss = ((pred * target)).sum() / (torch.sqrt((pred * pred).sum() * (target * target).sum()))
        return 1 - loss

    def forward(self, pred, gt_s):
        kl = self.KL_loss(pred, gt_s)

        cc = self.CC_loss(pred, gt_s)

        return kl, cc

