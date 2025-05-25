import math
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn

from config import parse_option
args, config = parse_option()

# ======================================================================================
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=True):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        # self.act = act_layer()
        self.act = getattr(nn, act_layer.split('.')[-1], None)()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError

# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

class MambaBlock(nn.Module, mamba_init):
    def __init__(
        self,
        d_model=96,
        d_out=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v2",
        # ======================
        forward_type="v2",
        channel_first=True,
        with_initial_state=False,
        # ======================
        zz_paths = None,
        zz_paths_rev = None,
        is_IMAM = True,
        is_CMAM = False,
        is_pretrain = False,
        **kwargs,  # {}
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        factory_kwargs = {"device": None, "dtype": None}

        with_initial_state = True

        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        assert d_inner % dt_rank == 0

        self.zz_paths_func = zz_paths
        self.zz_paths_rev_func = zz_paths_rev
        self.Modality_Num = config.DATA.Modality_Num

        self.is_pretrain = is_pretrain
        self.is_CMAM = is_CMAM

        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.forward = self.forward_Inter

        self.share_SSM = config.MODEL.SHARE_SSM
        if self.share_SSM:
            self.k_group = 1
        else:
            if self.is_CMAM:
                self.k_group = 4
            else:
                self.k_group = 4

        if self.is_CMAM:
            self.forward = self.forward_Cross
        else:
            self.forward = self.forward_Inter

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        self.x_proj = [nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj


        # ====================================================
        initialize = "v2"
        if initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((self.k_group, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, dt_rank)))
        elif initialize in ["v2"]:

            self.Ds = nn.Parameter(torch.ones((self.k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, dt_rank)))

        self.initial_state = None
        if with_initial_state: # True
            self.initial_state = nn.Parameter(torch.zeros((1, self.k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False) # (1, KR, D, N)


    def CrossScan(self, x, size, mask):

        out = x.new_empty(size=size)
        for i in range(self.k_group):
            out[:, i, :, :] = x[:, :, self.zz_paths[i].copy()]

        return out

    def CrossMerge(self, x, size, x_gate=None, is_CMAM=False, mask=None):
        B, K, D, H, W = size
        if not is_CMAM:
            x_rev = x.new_empty(size=(B, K, D, H * W))
            for i in range(self.k_group):
                x_rev[:, i, :, :] = x[:, i, :, self.zz_paths_rev[i]]
            x_rev = x_rev.view(B, K, D, H, W)
            if x_gate is not None:
                x_gate_expand = x_gate.view(B, 1, D, H, W).expand(B, K, D, H, W)
                x_rev = torch.mul(x_rev, x_gate_expand)
            out = torch.sum(x_rev, dim=1)

        else:
            x_rev = x.new_empty(size=(B, K, D, H * W * self.Modality_Num))
            for i in range(self.k_group):
                x_rev[:, i, :, :] = x[:, i, :, self.zz_paths_rev[i]]

            x_rev_list = []
            for j in range(self.Modality_Num):
                if x_gate is not None:
                    x_rev_list.append(torch.mul(torch.sum(x_rev[:, :, :, j * H * W: (j + 1) * H * W], dim=1),
                                                x_gate.view(B, D, H * W)))
                else:
                    x_rev_list.append(torch.sum(x_rev[:, :, :, j * H * W: (j + 1) * H * W], dim=1))

            out = torch.cat(x_rev_list, dim=0)
            out = out.view(B * self.Modality_Num, D, H, W)
        return out

    def forward_core(
            self,
            x: torch.Tensor = None,
            x_gate: torch.Tensor = None,
            # ==============================
            mask = None,
            # ==============================
            to_dtype=True,
            force_fp32=False,
            # ==============================
            ssoflex=True,
            # ==============================
            SelectiveScan=None,
            no_einsum=True,
            # ==============================
            chunk_size=16,
            # ==============================
            selective_scan_backend=None,
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # chunk_size = config.MODEL.VSSM.SSM_D_STATE
        N = config.MODEL.VSSM.SSM_D_STATE
        B, RD, H, W = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R

        x_proj_bias = getattr(self, "x_proj_bias", None)


        if self.is_CMAM:
            B = config.DATA.BATCH_SIZE
            L = L * self.Modality_Num

            x_list = []
            for i in range(self.Modality_Num):
                x_list.append(x[i * config.DATA.BATCH_SIZE: (i + 1) * config.DATA.BATCH_SIZE, :, :, :].view(B, RD, -1))
            x = torch.cat(x_list, dim=-1)
        else:
            x = torch.flatten(x, start_dim=2, end_dim=3)

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == config.MODEL.VSSM.SSM_D_STATE
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)


        if mask is not None:

            xs = self.CrossScan(x, size=(x.shape[0], self.k_group, x.shape[1], config.DATA.Modality_Num * config.DATA.PATCH_SIZE * config.DATA.PATCH_SIZE),mask=mask)  # Out[1]: torch.Size([512, 4, 128, 169]) --  in: torch.Size([1, 96, 56, 56]), CrossScan: models.csm_triton.CrossScanTriton, out: torch.Size([1, 4, 96, 3136])
            L = xs.shape[-1]
            xs = xs.permute(0, 3, 1, 2)

        else:
            xs = self.CrossScan(x, size=(x.shape[0], self.k_group, x.shape[1], x.shape[2]), mask=None) # B * 4 * C * L
            xs = xs.permute(0, 3, 1, 2)

        if self.share_SSM:
            ys = xs.new_empty(B, L, self.k_group, RD)
            for i in range(self.k_group):
                xs_s = xs[:, :, i, :]
                xs_s = xs_s.view(xs_s.shape[0], xs_s.shape[1], 1, xs_s.shape[2]) # B * L * 1 * C
                x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs_s, self.x_proj_weight)  # B * L * K * (R + N * 2)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
                xs_s = xs_s.contiguous().view(B, L, KR, D)
                dts = dts.contiguous().view(B, L, KR)
                Bs = Bs.contiguous().view(B, L, K, N)
                Cs = Cs.contiguous().view(B, L, K, N)
                if force_fp32:
                    xs_s, dts, Bs, Cs = to_fp32(xs_s, dts, Bs, Cs)

                As = -self.A_logs.to(torch.float).exp().view(KR)
                Ds = self.Ds.to(torch.float).view(KR, D)
                dt_bias = self.dt_projs_bias.view(KR)

                if force_fp32:
                    xs_s, dts, Bs, Cs = to_fp32(xs_s, dts, Bs, Cs)

                out_s, final_state = selective_scan_chunk_fn(
                    xs_s, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias,
                    initial_states=initial_state, dt_softplus=True, return_final_states=True,
                    backend=selective_scan_backend,
                )
                out_s = out_s.view(B, L, RD)

                ys[:, :, i, :] = out_s
        else:
            x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight) # B * L * K * (R + N * 2)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
            xs = xs.contiguous().view(B, L, KR, D)
            dts = dts.contiguous().view(B, L, KR)
            Bs = Bs.contiguous().view(B, L, K, N)
            Cs = Cs.contiguous().view(B, L, K, N)
            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            As = -self.A_logs.to(torch.float).exp().view(KR)
            Ds = self.Ds.to(torch.float).view(KR, D)
            dt_bias = self.dt_projs_bias.view(KR)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)


            ys, final_state = selective_scan_chunk_fn(
                xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias,
                initial_states=initial_state, dt_softplus=True, return_final_states=True,
                backend=selective_scan_backend,
            )
            ys = ys.view(B, L, K, RD)

        y: torch.Tensor = self.CrossMerge(ys.permute(0, 2, 3, 1), size=(B, self.k_group, RD, H, W), x_gate=x_gate, is_CMAM=self.is_CMAM,
                                              mask=mask)

        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_Inter(self, x: torch.Tensor, mask=None, x_gate=None, **kwargs):

        self.zz_paths = self.zz_paths_func(N=config.DATA.PATCH_SIZE)
        self.zz_paths_rev = self.zz_paths_rev_func(self.zz_paths)

        out = self.forward_core(x=x, x_gate=None, SelectiveScan=partial(selective_scan_fn, backend="mamba"))

        return out

    def forward_Cross(self, x: torch.Tensor, mask=None, x_gate=None, **kwargs):

        self.Modality_Num = x.shape[0] // config.DATA.BATCH_SIZE
        if mask is not None:
            self.zz_paths = self.zz_paths_func(N=config.DATA.PATCH_SIZE, Modality_Num=config.DATA.Modality_Num)
        else:
            self.zz_paths = self.zz_paths_func(N=config.DATA.PATCH_SIZE, Modality_Num=self.Modality_Num)
        self.zz_paths_rev = self.zz_paths_rev_func(self.zz_paths)

        out = self.forward_core(x=x, mask=mask, x_gate=x_gate, SelectiveScan=partial(selective_scan_fn, backend="mamba"))

        return out

class Text_Net(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text):
        x = self.token_embedding(text).type(torch.float32)

        x = x + self.positional_embedding.type(torch.float32)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(torch.float32)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.3):
        super(AttentionFusion, self).__init__()
        self.attention_layer = nn.Conv2d(feature_dim, 1, kernel_size=1, stride=1, padding=0)
        self.dropout_prob = dropout_prob

    def forward(self, inputs):
        attention_scores = []
        for x in inputs:
            score = self.attention_layer(x)
            attention_scores.append(score)

        attention_scores = torch.stack(attention_scores, dim=0)
        attention_scores = F.softmax(attention_scores, dim=0)
        attention_scores = F.dropout(attention_scores, p=self.dropout_prob, training=self.training)

        weighted_sum = 0
        for i, x in enumerate(inputs):
            weight = attention_scores[i]
            weighted_sum += weight * x

        return weighted_sum

# Modal-Aware Mamba  (MAMambaBlock)
class MAMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            output_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=True,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            is_pretrain=False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        def zigzag_cross_path(N, Modality_Num):
            def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
                path = []
                for i in range(N):
                    for j in range(N):
                        for k in range(Modality_Num):
                            col = j if i % 2 == 0 else N - 1 - j
                            if (i % 2 == 0):
                                depth = k if col % 2 == 0 else Modality_Num - 1 - k
                            else:
                                depth = Modality_Num - 1 - k if col % 2 == 0 else k
                            path.append((start_row + dir_row * i) * N + start_col + dir_col * col + depth * N * N)
                return path

            def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
                path = []
                for j in range(N):
                    for i in range(N):
                        for k in range(Modality_Num):
                            row = i if j % 2 == 0 else N - 1 - i
                            if (j % 2 == 0):
                                depth = k if row % 2 == 0 else Modality_Num - 1 - k
                            else:
                                depth = Modality_Num - 1 - k if row % 2 == 0 else k
                            path.append((start_row + dir_row * row) * N + start_col + dir_col * j + depth * N * N)
                return path

            paths = []
            for start_row, start_col, dir_row, dir_col in [
                (0, 0, 1, 1),
                (N - 1, N - 1, -1, -1),
            ]:
                paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
                paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

            for _index, _p in enumerate(paths):
                paths[_index] = np.array(_p)
            return paths

        def zigzag_path(N):
            def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
                path = []
                for i in range(N):
                    for j in range(N):
                        # If the row number is even, move right; otherwise, move left
                        col = j if i % 2 == 0 else N - 1 - j
                        path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
                return path

            def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
                path = []
                for j in range(N):
                    for i in range(N):
                        row = i if j % 2 == 0 else N - 1 - i
                        path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
                return path

            paths = []
            for start_row, start_col, dir_row, dir_col in [
                (0, 0, 1, 1),
                (N - 1, N - 1, -1, -1),
            ]:
                paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))  # 前向
                paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))  # 后向

            for _index, _p in enumerate(paths):
                paths[_index] = np.array(_p)
            return paths

        def raster_path(N):
            paths = []
            temp = np.arange(0, N * N).reshape(N, N)
            paths.append(temp.flatten())
            paths.append(temp.T.flatten())
            paths.append(temp.flatten()[::-1])
            paths.append(temp.T.flatten()[::-1])
            return paths

        def raster_path_cross(N, Modality_Num):
            paths_res = []
            paths_temp = [[] for _ in range(Modality_Num)]
            temps = [np.arange(N * N * i, N * N * (i + 1)).reshape(N, N) for i in range(Modality_Num)]
            for i, temp in enumerate(temps):  # f
                paths_temp[i].append(temp.flatten())
                paths_temp[i].append(temp.T.flatten())
            for path in zip(*paths_temp):
                path_ = np.ravel(np.column_stack(path))
                paths_res.append(path_)
                paths_res.append(path_[::-1])
            return paths_res

        def reverse_permut_np(permutation):
            n = len(permutation)
            reverse = np.array([0] * n)
            for i in range(n):
                reverse[permutation[i]] = i
            return reverse

        def reverse_path(path):
            return [reverse_permut_np(_) for _ in path]

        if config.MODEL.VSSM.SCAN_MODE == "Zigzag":
            self.zz_paths = zigzag_path
            self.zz_paths_cross = zigzag_cross_path
        elif config.MODEL.VSSM.SCAN_MODE == "Raster":
            self.zz_paths = raster_path
            self.zz_paths_cross = raster_path_cross
        else:
            Exception("Error_line_1396")

        self.zz_paths_rev = reverse_path
        self.zz_paths_cross_rev = reverse_path

        self.IMSM = nn.ModuleList([MambaBlock(
            d_model=hidden_dim,
            d_out=output_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            bias=False,
            # ==========================
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize=ssm_init,
            # ==========================
            forward_type=forward_type,
            channel_first=channel_first,
            # ===========================
            zz_paths=self.zz_paths,
            zz_paths_rev=self.zz_paths_rev,
            is_IMAM=True,
            is_CMAM=False,
            is_pretrain=is_pretrain,
        ) for _ in range(config.DATA.Modality_Num)])

        Linear = Linear2d if channel_first else nn.Linear
        self.LN1 = nn.ModuleList([norm_layer(hidden_dim) for _ in range(config.DATA.Modality_Num)])
        self.LN3 = nn.ModuleList([norm_layer(hidden_dim) for _ in range(config.DATA.Modality_Num)])
        self.act = nn.SiLU()

        self.Conv2D = nn.ModuleList([nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=ssm_conv,
                padding=(ssm_conv - 1) // 2,
                groups=hidden_dim // 2,
            ),
        ) for _ in range(config.DATA.Modality_Num)])

        self.out_proj = nn.ModuleList([Linear(hidden_dim, output_dim) for _ in range(config.DATA.Modality_Num)])
        self.AttFusion = nn.ModuleList([AttentionFusion(feature_dim=hidden_dim) for _ in range(config.DATA.Modality_Num)])

        if config.DATA.Modality_Num >= 2:
            self.LN2 = norm_layer(hidden_dim)
            self.Cat_Conv2D = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=ssm_conv,
                    padding=(ssm_conv - 1) // 2,
                    groups=hidden_dim,
                ),
            )
            self.Cat_proj = Linear(hidden_dim, hidden_dim)
            self.Cat_AttFusion = AttentionFusion(feature_dim=hidden_dim)
            self.Cat_IMSM = MambaBlock(
                d_model=hidden_dim,
                d_out=output_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                bias=False,
                # ==========================
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                # ===========================
                zz_paths=self.zz_paths,
                zz_paths_rev=self.zz_paths_rev,
                is_IMAM=True,
                is_CMAM=False,
                is_pretrain=is_pretrain,
            )

    def forward(self, input: tuple):
        x: torch.Tensor = input[0]
        mask: torch.Tensor  = input[1]
        B = config.DATA.BATCH_SIZE
        _, D, H, W = x.shape

        for i in range(config.DATA.Modality_Num):
            x_ = x[config.DATA.BATCH_SIZE * i: config.DATA.BATCH_SIZE * (i + 1)].clone()
            x_ = self.LN1[i](x_)
            x[config.DATA.BATCH_SIZE * i: config.DATA.BATCH_SIZE * (i + 1)] = x_

        if config.DATA.Modality_Num >= 2:
            x_cat = []
            for i in range(config.DATA.Modality_Num):
                x_cat.append(x[config.DATA.BATCH_SIZE * i : config.DATA.BATCH_SIZE * (i + 1)].clone())
            x_cat = self.Cat_AttFusion(x_cat)
            x_cat = self.LN2(x_cat)
            x_cat = self.Cat_Conv2D(x_cat)
            x_cat = self.Cat_IMSM(x_cat, mask=None, x_gate=None)
            x_cat = self.act(self.Cat_proj(x_cat))

        for i in range(config.DATA.Modality_Num):
            x_ = x[config.DATA.BATCH_SIZE * i : config.DATA.BATCH_SIZE * (i + 1)].clone()
            if config.DATA.Modality_Num >= 2:
                x_ = self.AttFusion[i]([x_, x_cat])
                x_ = self.LN3[i](x_)
            x_ = self.Conv2D[i](x_)
            x_ = self.IMSM[i](x=x_, mask=None, x_gate=None)
            x_ = self.act(self.out_proj[i](x_))
            x[config.DATA.BATCH_SIZE * i: config.DATA.BATCH_SIZE * (i + 1)] = x_

        return (x, None)

class DPMamba(nn.Module):
    def __init__(
            self,
            patch_size=3,
            in_chans=3,
            depths=[6, 9],
            dims=[96, 96],
            # =========================
            ssm_d_state=1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            # =========================
            drop_path_rate=0.2,
            patch_norm=True,
            norm_layer="LN2D",
            patchembed_version: str = "v2",
            use_checkpoint=False,
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # ==========================
            posembed=False,
            imgsize=9,
            is_pretrain=False,
            **kwargs,
    ):
        super().__init__()

        self.is_pretrain = is_pretrain

        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)

        ## ====================== patch embedding ==================================
        self.patch_embedding = nn.ModuleList()
        kernel_size = 3
        stride = 1
        padding = 1
        dims = []
        if (config.DATA.DATASET_NAME == "Trento"):
            if (config.DATA.N_PCA != -1):
                self.patch_embedding.append(nn.Sequential(nn.Conv2d(in_channels=config.DATA.N_PCA, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size, stride=stride, padding=padding),
                                                nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                                                nn.ReLU(inplace=True),
                                                ))
                dims.append(config.DATA.N_PCA)
            else:
                self.patch_embedding.append(nn.Sequential(
                    nn.Conv2d(in_channels=63, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size,stride=stride, padding=padding),
                    nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                    nn.ReLU(inplace=True),
                    ))
                dims.append(63)
            self.patch_embedding.append(nn.Sequential(nn.Conv2d(in_channels=1, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size, stride=stride, padding=padding),
                                            nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                                            nn.ReLU(inplace=True)))
            dims.append(1)

        elif (config.DATA.DATASET_NAME == "Houston"):
            if (config.DATA.N_PCA != -1):
                self.patch_embedding.append(nn.Sequential(nn.Conv2d(in_channels=config.DATA.N_PCA, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size, stride=stride, padding=padding),
                                                nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                                                nn.ReLU(inplace=True),
                                                ))
                dims.append(config.DATA.N_PCA)
            else:
                self.patch_embedding.append(nn.Sequential(
                    nn.Conv2d(in_channels=144, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size,stride=stride, padding=padding),
                    nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                    nn.ReLU(inplace=True),
                    ))
                dims.append(144)
            self.patch_embedding.append(nn.Sequential(nn.Conv2d(in_channels=1, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size, stride=stride, padding=padding),
                                            nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                                            nn.ReLU(inplace=True)))
            dims.append(1)

        elif (config.DATA.DATASET_NAME == "Augsburg_HSI_SAR"):
            if (config.DATA.N_PCA != -1):
                self.patch_embedding.append(nn.Sequential(nn.Conv2d(in_channels=config.DATA.N_PCA, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size, stride=stride, padding=padding),
                                                nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                                                nn.ReLU(inplace=True),
                                                ))
                dims.append(config.DATA.N_PCA)
            else:
                self.patch_embedding.append(nn.Sequential(
                    nn.Conv2d(in_channels=180, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size,stride=stride, padding=padding),
                    nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                    nn.ReLU(inplace=True),
                    ))
                dims.append(180)
            self.patch_embedding.append(nn.Sequential(nn.Conv2d(in_channels=4, out_channels=config.MODEL.HIDDEN_DIM_IMAM, kernel_size=kernel_size, stride=stride, padding=padding),
                                            nn.BatchNorm2d(config.MODEL.HIDDEN_DIM_IMAM),
                                            nn.ReLU(inplace=True)))
            dims.append(4)

        else:
            raise Exception(f"{config.DATA.DATASET_NAME} is Wrong.")


        if not self.is_pretrain:
            self.prompt_spe_m = nn.ParameterList([nn.Parameter(torch.randn(config.MODEL.HIDDEN_DIM_IMAM, config.DATA.PATCH_SIZE, config.DATA.PATCH_SIZE)) for dim in dims])

        config.defrost()
        config.DATA.PATCH_SIZE = math.floor((config.DATA.IMG_SIZE - kernel_size + 2 * padding) / stride) + 1
        config.freeze()
        self.num_patches = (config.DATA.PATCH_SIZE) ** 2


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.Modality_Aware_Mamba = self._make_layer_(
                    drop_path=dpr[0: depths[0]],
                    norm_layer=norm_layer,
                    channel_first=self.channel_first,
                    # =================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =================
                    use_checkpoint=use_checkpoint,
                    post_norm=False,
                    # ==================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate = mlp_drop_rate,
                    gmlp = gmlp,
                    # ==================
                    is_pretrain=is_pretrain,
                )

        self.project = nn.Sequential(OrderedDict(
            norm=norm_layer(config.MODEL.VSSM.EMBED_DIM[0][-1] * config.DATA.Modality_Num),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(config.MODEL.VSSM.EMBED_DIM[0][-1] * config.DATA.Modality_Num, 512),
        ),)

        self.norm = nn.ModuleList([norm_layer(config.MODEL.HIDDEN_DIM_IMAM) for i in range(config.DATA.Modality_Num)])
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_layer_(
            drop_path=[0.1, 0.1],
            norm_layer=nn.LayerNorm,
            channel_first=True,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            use_checkpoint = False,
            post_norm = False,
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            is_pretrain=False,
            **kwargs,
    ):

        depth = len(drop_path)
        dims = config.MODEL.VSSM.EMBED_DIM[0]
        blocks = []
        for d in range(depth):
            if d == len(dims) - 1:
                o_dim = dims[d]
            else:
                o_dim = dims[d + 1]
            i_dim = dims[d]

            blocks.append(MAMambaBlock(
                hidden_dim=i_dim,
                output_dim=o_dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                # =============================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # ============================
                use_checkpoint=use_checkpoint,
                post_norm = post_norm,
                # =============================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # ==============================
                is_pretrain=is_pretrain,
            ))

        return nn.Sequential(*blocks)

    def generate_binary_combinations(self, n):
        combinations = []
        for i in range(1, 2 ** n):
            binary_combination = [(i >> j) & 1 for j in range(n - 1, -1, -1)]
            combinations.append(binary_combination)
        return torch.tensor(combinations)

    def forward(self, x, mask=None):

        config.defrost()
        config.DATA.BATCH_SIZE = x[0].shape[0]
        config.freeze()

        ## ===============  patch embedding  ===================
        for i, conv_ in enumerate(self.patch_embedding):
            x[i] = self.norm[i](conv_(x[i]))

        ## ===============  Prompt Inject  ===================
        if mask is not None:
            B = config.DATA.BATCH_SIZE
            _, D, H, W = x[0].shape

            for i in range(config.DATA.Modality_Num):
                x_new = []
                if self.is_pretrain:
                    img_dim = x[i].shape[-3]
                    prompt = torch.zeros(img_dim, H, W).cuda()
                else:
                    prompt = self.prompt_spe_m[i]

                for j, mask_code in enumerate(mask[:, i]):
                    if mask_code == 0:
                        x_new.append(prompt)
                    else:
                        x_new.append(x[i][j])
                x[i] = torch.stack(x_new, dim=0)

        x = torch.cat(x, dim=0)

        x, _ = self.Modality_Aware_Mamba((x, None))

        if config.DATA.Modality_Num >= 2:
            x_fused = []
            for i in range(config.DATA.Modality_Num):
                x_fused.append(x[config.DATA.BATCH_SIZE * i: config.DATA.BATCH_SIZE * (i + 1)])
            x_fused = torch.cat(x_fused, dim=1)
        else:
            x_fused = x

        project_out = self.project(x_fused)

        out_features = []
        out_features.append(project_out)

        loss = [None, None]
        return [out_features, loss]

