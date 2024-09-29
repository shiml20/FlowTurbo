# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# SiT：https://github.com/willisma/SiT/blob/main/models.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp ,Attention
import einops
from diffusers.models import AutoencoderKL


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core FlowTurbo Model                                #
#################################################################################

class FlowTurboBlock(nn.Module):
    """
    A FlowTurbo block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of FlowTurbo.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class VelocityPredictor(nn.Module):
    """
    Velocity Predictor model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        in_channels_scale=1
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels*in_channels_scale, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            FlowTurboBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in FlowTurbo blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, v, y, force_drop_ids=None):
        
        """
        Forward pass of FlowTurbo.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        if v is not None:
            x = torch.concatenate([x, v], dim=1)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training, force_drop_ids)    # (N, D)
        c = t + y                                # (N, D)

        for idx, block in enumerate(self.blocks):
            x = block(x, c)                       # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                    # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
            
        return x

    def forward_with_cfg(self, x, t, v, y, cfg_scale, force_drop_ids=None):
        """
        Forward pass of FlowTurbo, but also batches the uncondional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, v, y, force_drop_ids)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return torch.cat([eps, rest], dim=1)


class VelocityRefinerBlock(nn.Module):
    def __init__(self,
                input_size=32,
                patch_size=2,
                in_channels=4,
                hidden_size=1152,
                depth=1,
                num_heads=16,
                mlp_ratio=4.0,
                class_dropout_prob=0.1,
                num_classes=1000,
                learn_sigma=True,
                num_steps = 10
   ) -> None:
        super().__init__()


        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels*2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
            FlowTurboBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])


        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.t = torch.linspace(0, 1, num_steps)
        self.delta_t = self.t[1] - self.t[0]


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def _forward(self, x, t, y, cfg_scale, v):
        t = torch.ones(x[0].size(0)).to(x.device) * t if isinstance(x, tuple) else torch.ones(x.size(0)).to(x.device) * t
       
        half = x[: len(x) // 2]
        x = torch.cat([half, half], dim=0)

        if v is not None:
            x = torch.concatenate([x, v], dim=1)

        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for idx, block in enumerate(self.blocks):
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)

        eps, rest = x[:, :3], x[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        delta_v = torch.cat([eps, rest], dim=1)
        
        return delta_v

    def forward(self, input):
        x, i, v, y, cfg_scale= input[0], input[1], input[2], input[3], input[4]
        print(i)
        y_ori = y
        xi = x

        d_i = self._forward(x, self.t[i], y, cfg_scale, v) + v
        x_tilde_i_plus_i = xi + self.delta_t * d_i
        d_i_plus_1 = self._forward(x_tilde_i_plus_i, self.t[i+1], y, cfg_scale, v) + d_i
        xi = xi + 1/2 * self.delta_t * (d_i + d_i_plus_1)

        return xi, i + 1, d_i, y_ori, cfg_scale


class VelocityBlock(nn.Module):
    def __init__(self,
                input_size=32,
                patch_size=2,
                in_channels=4,
                hidden_size=1152,
                depth=28,
                num_heads=16,
                mlp_ratio=4.0,
                class_dropout_prob=0.1,
                num_classes=1000,
                learn_sigma=True,
                num_steps=10,
                use_cache=False
   ) -> None:
        super().__init__()
        self.use_cache = use_cache
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
            FlowTurboBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.t = torch.linspace(0, 1, num_steps)
        self.delta_t = self.t[1] - self.t[0]

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _forward(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        x = torch.cat([half, half], dim=0)
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for idx, block in enumerate(self.blocks):
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        eps, rest = x[:, :3], x[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        v = torch.cat([eps, rest], dim=1)
        return v

    def forward(self, input):
        x, i, d_i, y, cfg_scale= input[0], input[1], input[2], input[3], input[4]
        y_ori = y

        if d_i is not None and self.use_cache:
            x_ori = x
        else:
            x_ori = x
            t = self.t[i]
            t = torch.ones(x[0].size(0)).to(x.device) * t if isinstance(x, tuple) else torch.ones(x.size(0)).to(x.device) * t
            d_i = self._forward(x, t, y, cfg_scale)

        x_tilde_i_plus_i = x + self.delta_t * d_i
        x = x_tilde_i_plus_i      
        t = self.t[i+1]
        t = torch.ones(x[0].size(0)).to(x.device) * t if isinstance(x, tuple) else torch.ones(x.size(0)).to(x.device) * t
        d_i_plus_1 = self._forward(x, t, y, cfg_scale)
        xi =  x_ori + 1/2 * self.delta_t * (d_i + d_i_plus_1)

        return xi, i + 1, d_i_plus_1, y_ori, cfg_scale, d_i


class FlowTurboAssemble(nn.Module):
    def __init__(self, 
                input_size=32,
                patch_size=2,
                in_channels=4,
                hidden_size=1152,
                depth=28,
                num_heads=16,
                mlp_ratio=4.0,
                class_dropout_prob=0.1,
                num_classes=1000,
                learn_sigma=True,
                predictor_ckpt = 'predictor_ckpt_path',
                refiner_ckpt = 'refiner_ckpt_path',
                vae_ckpt = 'stabilityai/sd-vae-ft-ema',
                N_H=2,
                N_P=4,
                N_R=3,
                SAC=False
                ) -> None:
        super().__init__()

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_steps = N_H + N_P + N_R + 1

        self.velocity_heun = VelocityBlock(input_size=input_size,
                                                patch_size=patch_size,
                                                in_channels=in_channels,
                                                hidden_size=hidden_size,
                                                depth=28,
                                                num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                class_dropout_prob=class_dropout_prob,
                                                num_classes=num_classes,
                                                learn_sigma=learn_sigma,
                                                num_steps=self.num_steps,
                                                use_cache=False)

        self.velocity_pseudo = VelocityBlock(input_size=input_size,
                                                patch_size=patch_size,
                                                in_channels=in_channels,
                                                hidden_size=hidden_size,
                                                depth=28,
                                                num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                class_dropout_prob=class_dropout_prob,
                                                num_classes=num_classes,
                                                learn_sigma=learn_sigma,
                                                num_steps=self.num_steps,
                                                use_cache=True)

        self.velocity_refiner = VelocityRefinerBlock(input_size=input_size,
                                                patch_size=patch_size,
                                                in_channels=in_channels,
                                                hidden_size=hidden_size,
                                                depth=1,
                                                num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                class_dropout_prob=class_dropout_prob,
                                                num_classes=num_classes,
                                                learn_sigma=learn_sigma,
                                                num_steps=self.num_steps)

        self.load_ckpt(self.velocity_heun, predictor_ckpt)
        self.load_ckpt(self.velocity_pseudo, predictor_ckpt)
        self.load_ckpt(self.velocity_refiner, refiner_ckpt)
        self.vae = AutoencoderKL.from_pretrained(vae_ckpt)

        if SAC:
            print("Using SAC. The first run requires compilation, please wait.")
            self.velocity_heun = torch.compile(self.velocity_heun)
            self.velocity_pseudo = torch.compile(self.velocity_pseudo)
            self.velocity_refiner = torch.compile(self.velocity_refiner)
            self.vae.decoder = torch.compile(self.vae.decoder)
            # print("Compiled.")


        self.heun_seq = nn.Sequential(
            *[self.velocity_heun for _ in range(N_H)]
        )

        self.pseudo_seq = nn.Sequential(
            *[self.velocity_pseudo for _ in range(N_P)]
        )

        self.refine_seq = nn.Sequential(
            *[self.velocity_refiner for _ in range(N_R)]
        )


    def find_model(self, model_name):
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "model" in checkpoint: 
            checkpoint = checkpoint["model"]
        return checkpoint

    def load_ckpt(self, model, ckpt_path):
        checkpoint_dict = self.find_model(ckpt_path)
        model_dict = checkpoint_dict
        model.load_state_dict(model_dict)


    def forward(self, x, **model_kwargs):

        y = model_kwargs['y']
        cfg_scale = model_kwargs['cfg_scale']
        i = torch.tensor(0, device=x.device)
        heun_output = self.heun_seq([x, i, None, y, cfg_scale])
        # x, i, v, y, cfg_scale
        pseudo_input = [heun_output[0], heun_output[1], heun_output[2], heun_output[3], heun_output[4]]
        pseudo_output = self.pseudo_seq(pseudo_input)
        # x, i, v, y, cfg_scale
        refiner_input = [pseudo_output[0], pseudo_output[1], pseudo_output[2], pseudo_output[3], pseudo_output[4]]
        refiner_output = self.refine_seq(refiner_input)
        img = self.vae.decode(refiner_output[0].chunk(2, 0)[0] / 0.18215).sample

        return img, refiner_output[0]

        
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
