from kornia.filters import filter2d
from torch import nn
import torch
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer,activation_function="relu", upsample=True):
        super().__init__()

        # Define layers individually so you can call them one-by-one


        UpSample = nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear') if not upsample else nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        if activation_function == "relu":
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)), # It seems like including 3d conv, but it works as 2d conv, while kernel_size=(3, 3, 1)
                norm_layer,
                nn.ReLU(),
                UpSample,
            )
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0))
            )


    def forward(self, x):
        for k, layer in enumerate(self.block):
            x = layer(x)
        return x
