from model.utils.projection import Erp2Tangent, Tangent2Erp
from model.decoder_blocks import DecoderBlock
from model.VSTCA import VSTCA

from dotmap import DotMap
from torch import nn

from einops import rearrange

import torch
import torch.nn.functional as F


SPH_EMB_SIZE = 56

import clip
print("CLIP loaded from:", clip.__file__)
def SimEst(visual_feat, text_feat):
    text_embed_norm = F.normalize(text_feat.float(), p=2, dim=-1)  # [B, 1024]
    frames_norm = F.normalize(visual_feat, p=2, dim=-1)
    cos_sim = torch.einsum('bd,btpd->btp', text_embed_norm, frames_norm)

    min_vals = cos_sim.min(dim=2, keepdim=True)[0]  # [B, 8, 1]
    max_vals = cos_sim.max(dim=2, keepdim=True)[0]  # [B, 8, 1]
    weights = (cos_sim - min_vals) / (max_vals - min_vals + 1e-6)

    return weights.unsqueeze(-1)

def extract_resnet_features(visual, image_tensor):
    x = visual.conv1(image_tensor)
    x = visual.bn1(x)
    x = visual.relu1(x)
    x = visual.conv2(x)
    x = visual.bn2(x)
    x = visual.relu2(x)
    x = visual.conv3(x)
    x = visual.bn3(x)
    x = visual.relu3(x)
    x = visual.avgpool(x)

    x = visual.layer1(x)
    x1 = visual.layer2(x)  # [B, 512, 28, 28]
    x2 = visual.layer3(x1)  # [B, 1024, 14, 14]
    x3 = visual.layer4(x2) # [B, 2048, 7, 7]
    x_global = visual.attnpool(x3)

    return x_global.float(), [x3.float(), x2.float(), x1.float()]

def extract_text_features(clip_text, text_tokens):
    text_global_features =  clip_text.encode_text(text_tokens)

    # Get transformer outputs
    x = clip_text.token_embedding(text_tokens).type(clip_text.dtype)  # [1, 77, 512]
    x = x + clip_text.positional_embedding.type(clip_text.dtype)
    x = x.permute(1, 0, 2)  # [77, 1, 512]

    # Process through transformer
    all_layer_outputs = []
    for layer in clip_text.transformer.resblocks:
        x = layer(x)
        all_layer_outputs.append(x.permute(1, 0, 2))  # [1, 77, 512]

    # Final processing
    text_features2 = x.permute(1, 0, 2)  # [1, 77, 512]

    x = clip_text.ln_final(text_features2).type(clip_text.dtype)

    token_mask = (text_tokens != 0) & \
                 (text_tokens != 49406) & \
                 (text_tokens != 49407)

    # Get per-token features (masked)
    text_local_features = x @ clip_text.text_projection  # [1, 77, 1024]

    return text_global_features, text_local_features, token_mask


class TSalV360(nn.Module):
    def __init__(self, config: DotMap = None,erp_size=(960, 1920)):
        super().__init__()

        network_config = config.network
        tangent_config = config.tangent_images

        self.nrows = tangent_config.nrows
        self.fov = tangent_config.fov
        self.patch_size_e2p = tangent_config.patch_size.e2p[0]

        self.clip, _ = clip.load("RN50", device='cuda')

        #manually for each of the 3 extracted layers
        self.vis_dims = [2048,1024,512]
        self.text_dims = [1024,1024,1024]
        self.emb_dims = [2048, 1024, 512]
        self.resnet_norm_layer = nn.ModuleList([
            nn.LayerNorm([2048, 7, 7]),
            nn.LayerNorm([1024, 14, 14]),
            nn.LayerNorm([512, 28, 28])
        ])


        nrows = tangent_config.nrows
        patch_size = tangent_config.patch_size.e2p[0]
        fov = tangent_config.fov
        self.number_of_tangent_images= tangent_config.npatches
        self.local_visual_features_hw = [7,14,28]

        tangent_config = {
                'nrows': nrows,
                'patch_size': patch_size,
                'fov': fov
            }
        shift = False
        self.E2T = Erp2Tangent(erp_size=erp_size, fov=fov,nrows=nrows, patch_size=patch_size,shift=shift)
        self.T2E = Tangent2Erp(erp_size=(240, 480), fov=tangent_config['fov'], nrows=tangent_config['nrows'],shift=shift, patch_size=(56, 56))
        self.sph_coords = []
        self.sph_coord_dim = []
        # Extract 3 different spherical coordinates for each of the 3 extracted local visual layers
        for i in range(3):
            self.sph_coords.append(
                F.adaptive_avg_pool2d(
                    Erp2Tangent(erp_size, fov=tangent_config['fov'], nrows=tangent_config['nrows'], shift=shift,
                              patch_size=SPH_EMB_SIZE).get_spherical_embeddings(),
                    (self.local_visual_features_hw[i], self.local_visual_features_hw[i]))
            )
            self.sph_coord_dim.append(5*self.local_visual_features_hw[i]*self.local_visual_features_hw[i])

        # Decoder
        self.set_tangent_decoder(config)

        self.down = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Out size: [b t 512 1 1]

        config_transformer = network_config.VSTCA

        self.VSTCA = nn.ModuleList()

       #Create 3 different VSTCA modules for each of the 3 extracted layers
        for crr in range(3):
            transformer_hparams = {
                'emb_dim': self.emb_dims[crr],
                'vis_dim': self.vis_dims[crr],
                'text_dim': self.text_dims[crr],
                'sph_input_dim': self.sph_coord_dim[crr],
                'depth': config_transformer.depth,
                'num_heads': config_transformer.num_heads,
                'mlp_mult': config_transformer.mlp_dim,
                'ffdropout': config_transformer.ff_dropout,
                'attn_dropout': config_transformer.attn_dropout,
            }

            transformer = VSTCA(**transformer_hparams)
            self.VSTCA.append(transformer)


    def forward(self, video_frames, text_input):
        text_tokens = clip.tokenize(text_input).to("cuda")
        with torch.no_grad():

            video_tensor = video_frames.float().cuda()
            x_tang = self.E2T.project_clip(video_tensor)  # BS, F=8, C=3, H=224, W=224, T=18
            BS, F, _, _, _, T = x_tang.shape
            x_tang = rearrange(x_tang, 'b f c h w t -> (b f t) c h w', b=BS, f=F)

            visual_global_features, visual_local_features = extract_resnet_features(self.clip.visual, x_tang.half())

            text_global_features, text_local_features, token_mask  = extract_text_features(self.clip,text_tokens)


            x_global_features = rearrange(visual_global_features, '(b f t) c -> b f t c', b=BS, f=F)
            sim_weights = SimEst(x_global_features,text_global_features)


        input_to_decoder = []
        for i, layer in enumerate(visual_local_features):
            #print("layer shape",layer.shape)
            _, C, H, W = layer.shape

            with torch.no_grad():
                sph_coords = self.sph_coords[i].to(x_tang.device)  # T, 5, 14, 14
                sph_emb = rearrange(sph_coords, 't c h w -> t (c h w)')  # T, 5*14*14=980


            layer = rearrange(layer, '(b f t) c h w -> b f t c h w', b=BS, f=F)

            down_layer = self.down(layer)
            down_layer = rearrange(down_layer, 'b f t c h w -> b f t (c h w)', b=BS, f=F)
            down_layer =  down_layer*sim_weights


            out = self.VSTCA[i](down_layer,text_local_features,token_mask,sph_emb)

            out = rearrange(out[:, -1], 'b t d -> b d 1 1 t')
            layer = layer[:, -1]

            layer = rearrange(layer, 'b t c h w -> (b t) c h w ', b=BS, t=T, c=C, h=H)
            layer = self.resnet_norm_layer[i](layer)

            layer = rearrange(layer, '(b t) c h w -> b c h w t', b=BS, t=T)
            input_to_decoder.append(layer+out)

        for i in range(self.decoder_length):
            if i==0: #first input to the decoder
                dec_block_output = self.decoders[i](input_to_decoder[i])
            elif i<=2: #hierachical skip connection (includes concatenation at channel level)
                output = torch.cat([dec_block_output, input_to_decoder[i]], dim=1)
                dec_block_output = self.decoders[i](output)
            else: #rest of the decoder blocks
                dec_block_output = self.decoders[i](dec_block_output)

        decoder_out = self.T2E(dec_block_output)
        pred = nn.functional.interpolate(decoder_out, size=(480, 960), mode='bilinear', align_corners=False)

        return pred

    def set_tangent_decoder(self, config):
        """
        Full tangent decoder.
        input: Tangent Feature Maps [B, C, H, W=, T], initial input [B,2048,7,7,18]
        """

        t = self.number_of_tangent_images
        norm_layers = [
            nn.LayerNorm([1024, 7, 7, t]),
            nn.LayerNorm([512, 14, 14, t]),
            nn.LayerNorm([512, 28, 28, t]),
            nn.LayerNorm([128, 56, 56, t]),
            nn.LayerNorm([1,56,56,t])
            #nn.LayerNorm([1, 56, 56, t])
        ]

        self.decoders = nn.Sequential(
            DecoderBlock(2048, 1024, norm_layers[0], activation_function = "relu", upsample=True),
            DecoderBlock(2048, 512, norm_layers[1], activation_function = "relu", upsample=True),
            DecoderBlock(1024, 512, norm_layers[2], activation_function = "relu", upsample=True),
            DecoderBlock(512, 128, norm_layers[3], activation_function="relu", upsample=False),
            DecoderBlock(128, 1, norm_layers[4], activation_function = "Sigmoid", upsample=False),nn.Sigmoid())
            #DecoderBlock(128, 1, norm_layers[4], activation_function="Sigmoid", upsample=False))

        self.decoder_length = len(self.decoders)



