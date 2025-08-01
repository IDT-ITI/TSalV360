import torch
import torch.nn as nn
from einops import rearrange, repeat

class VSTCA(nn.Module):
    def __init__(self, emb_dim,vis_dim,text_dim, sph_input_dim, depth=6, num_heads=4, ffdropout=0.0, attn_dropout=0.0, mlp_mult=4.0,
                 **kwargs):
        super().__init__()
        # print('VSTA Depth:', depth)

        self.layer = nn.ModuleList(
            [VSCTABlock(emb_dim,vis_dim,text_dim, num_heads=num_heads, drop=ffdropout, attn_drop=attn_dropout, mlp_ratio=mlp_mult,
                       **kwargs) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)

        self.sph_posemb = nn.Linear(sph_input_dim, vis_dim)
        self.tmp_posemb = nn.Parameter(torch.randn(1, 8, vis_dim))

    def forward(self, x, text_features, token_mask, sph, *args):
        """
        x: img embeddings [BS, F, T, emb_dim]
        sph: flattened spherical coordinates. [T, 980]
        """
        spatial_posemb = rearrange(self.sph_posemb(sph), 't d -> 1 1 t d')  # 1, 1, 18, 512

        temporal_posemb = repeat(self.tmp_posemb, '1 f d -> 1 f 1 d')  # 1, 8, 1, 512
        x = x + spatial_posemb + temporal_posemb

        for i, layer_block in enumerate(self.layer):
            x = layer_block(x,text_features,token_mask, *args)
        x = self.norm(x)
        return x
class VSCTABlock(nn.Module):
    def __init__(self, dim, vis_dim,text_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Temporal Attention Layers
        self.temp_norm_layer = norm_layer(vis_dim)
        self.temp_attn = Attention(dim, vis_dim, pos_emb='temporal', num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop=attn_drop, proj_drop=drop)

        # Spatial Attention
        self.spatial_norm_layer = norm_layer(vis_dim)
        self.spatial_attn = Attention(dim, vis_dim, pos_emb='spatial', num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      attn_drop=attn_drop, proj_drop=drop)


        self.cross_attn_norm_layer = norm_layer(vis_dim)
        self.cross_attn = CrossAttention(dim,vis_dim,text_dim,num_heads=num_heads,attn_dropout=attn_drop)


        # Final Feed-Forward-Network
        self.FFN = nn.Sequential(
            norm_layer(vis_dim),
            Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop),
        )


    def forward(self, x,text_features, token_mask, *args):
        B, F, T, D = x.shape  # number of tangent images.

        x = self.temp_attn(self.temp_norm_layer(x),"no", 'B F T D', '(B T) F D', T=T) + x
        x = self.spatial_attn(self.spatial_norm_layer(x),"yes", 'B F T D', '(B F) T D', F=F) + x
        x = self.cross_attn(self.cross_attn_norm_layer(x), text_features, token_mask) + x
        x = self.FFN(x) + x
        return x


class Attention(nn.Module):
    def __init__(self, dim,vis_dim, pos_emb, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, vis_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.QKV = nn.Linear(vis_dim, dim * 3, bias=qkv_bias)


    def forward(self, x, spatial, einops_from, einops_to, **einops_dims):

        h = self.num_heads
        q, k, v = self.QKV(x).chunk(3, dim=-1)

        # Divide heads
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=h), (q, k, v))
        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))

        attn = self.attn_drop((q @ k.transpose(-2, -1)).softmax(dim=-1))

        x1 = attn @ v
        x = rearrange(x1, f'{einops_to} -> {einops_from}', **einops_dims)

        # Merge heads
        x = rearrange(x, '(b h) ... d -> b ... (h d)', h=h)
        x = self.proj_drop(self.proj(x))
        return x

class CrossAttention(nn.Module):

    def __init__(self, dim=512,vis_dim=512,text_dim=512, num_heads=4, attn_dropout=0.5, dropout_1=0.2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout_res = nn.Dropout(dropout_1)

        self.q_proj = nn.Linear(vis_dim, dim, bias=False)
        self.to_k = nn.Linear(text_dim, dim, bias=False)
        self.to_v = nn.Linear(text_dim, dim, bias=False)
        self.proj = nn.Linear(dim, vis_dim, bias=False)
        self.text_norm = nn.LayerNorm(text_dim, eps=1e-6)
        #self.scale_2 = nn.Parameter(torch.zeros(1))

    def forward(self, visual_feat, text_feat, token_mask):
        """
        visual_feat: [B, T, S, D]
        text_feat: [B, L, D]
        token_mask: [B, L] (bool)
        """

        B, T, S, D = visual_feat.shape
        L = text_feat.size(1)
        h = self.num_heads
        dh = D // h
        scale = self.scale  # usually 1 / sqrt(dh)
        text_feat = self.text_norm(text_feat.float())
        # Project to Q, K, V
        q = self.q_proj(visual_feat)  # [B, T, S, D]
        k = self.to_k(text_feat)  # [B, L, D]
        v = self.to_v(text_feat)  # [B, L, D]

        # Rearrange for multi-head attention: split heads and flatten
        # Using einops style manually here:
        q = q.view(B, T * S, h, dh).permute(0, 2, 1, 3).reshape(B * h, T * S, dh)  # [B*h, T*S, dh]
        k = k.view(B, L, h, dh).permute(0, 2, 1, 3).reshape(B * h, L, dh)  # [B*h, L, dh]
        v = v.view(B, L, h, dh).permute(0, 2, 1, 3).reshape(B * h, L, dh)  # [B*h, L, dh]

        q = q * scale

        # Attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # [B*h, T*S, L]

        # Mask: expand token_mask to [B, 1, 1, L] then flatten batch*heads
        mask = token_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        mask = mask.expand(B, h, T * S, L).reshape(B * h, T * S, L)  # [B*h, T*S, L]

        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Attention output
        out = torch.bmm(attn, v)  # [B*h, T*S, dh]

        # Reshape back and concat heads
        out = out.view(B, h, T * S, dh).permute(0, 2, 1, 3).reshape(B, T, S, D)  # [B, T, S, D]

        # Final projection
        out = self.proj(out)

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        #drop = 0.3
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layers(x)