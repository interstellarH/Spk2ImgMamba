import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import repeat
from typing import Optional, Callable
from functools import partial

import torch.nn.functional as F

################Basic Model########################

class BasicModel(nn.Module): # base class
    def __init__(self):
        super().__init__()
    
    ## Tools functions for neural networks
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def num_parameters(self):
        return sum([p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])
    
    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

################Representation########################

def TFP(spk, channel_step=1):
    dim = spk.size(1) // (2*channel_step) # output_dim
    rep_spk = torch.mean(spk, dim=1).unsqueeze(1)
    for i in range(1, dim):
        rep_spk = torch.cat((rep_spk, torch.mean(spk[:, i*channel_step : -i*channel_step, :, :], 1).unsqueeze(1)), 1)
    return rep_spk 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_channles, use_1x1conv=True, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, num_channles, kernel_size=3, stride=strides, padding=1).cuda()
        self.conv2 = nn.Conv2d(
            num_channles, num_channles, kernel_size=3, padding=1).cuda()
        if use_1x1conv:
            self.conv3=nn.Conv2d(
                in_channels, num_channles,kernel_size=1, stride=strides).cuda()
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channles).cuda()
        self.bn2=nn.BatchNorm2d(num_channles).cuda()
        self.relu=nn.ReLU(inplace=True).cuda()
    def forward(self,x):
        y= self.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3:
            x=self.conv3(x)
        y+=x
        return self.relu(y)
            

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_depth=3,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_depth = d_depth
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_depth * self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.dwconv = nn.ModuleList()
        for _ in range(self.d_depth): # layer_num
            self.dwconv.append(nn.Conv2d(self.d_inner, self.d_inner, d_conv, 1, (d_conv-1)//2, groups=self.d_inner, bias=conv_bias))

        self.merge_conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, 1, (d_conv-1)//2)
        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

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
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_core(self, x: torch.Tensor, dir: int):
        # print("in forward_core, x.shape:", x.shape)
        B, C, H, W = x.shape
        L = H * W

        # original and transposed version
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # flipped version
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # [B, 4, -1, L]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, 4, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, 4, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, 4, -1, L)
        Cs = Cs.float().view(B, 4, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, 4, -1, L)
        assert out_y.dtype == torch.float

        if dir == 0:
            return out_y[:, 0]
        elif dir == 1:
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            return wh_y
        elif dir == 2:
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            return inv_y[:, 0]
        elif dir == 3:
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            return invwh_y
        elif dir == 4: # all directions
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            return out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        else:
            raise ValueError("Invalid direction value. Must be one of [0, 1, 2, 3].")

    def forward(self, x: torch.Tensor, dir: int = 0, **kwargs):
        B, H, W, C = x.shape

        x = self.in_proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        initial_x = x
        # print("initial_x.shape:", initial_x.shape)
        x = self.act(self.conv2d(x))

        # channel splitting
        split_x = x.chunk(self.d_depth, dim=1) # tuple
        # print("x, x1.shape:", x.shape, split_x[0].shape)

        # downsampling via pool integrated with dw-conv 
        x_list = [self.dwconv[0](split_x[0])]
        for i in range(1, self.d_depth):
            x_list.append(self.dwconv[i](nn.AdaptiveAvgPool2d((H//(2**i), W//(2**i)))(split_x[i])))
        # print("after downsampling, xs shape: ", x_list[0].shape, x_list[1].shape,)
        
        # ssm + upsampling
        for j in range(self.d_depth):
            if j==0:
                x_list[j] = self.forward_core(x_list[j], dir=4) 
            else:
                x_list[j] = F.interpolate(self.forward_core(x_list[j], dir=4), scale_factor=4**j, mode='linear', align_corners=True)
        # print("after ssm and upsampling, xs shape:", x_list[0].shape, x_list[1].shape)

        # Merge within MSD
        merge_x = torch.sum(torch.stack(x_list), dim=0)
        # print("merge_x.shape", merge_x.shape)

        final_x = initial_x.view(B, self.d_inner, -1).contiguous() + self.merge_conv(merge_x)
        # print("after 1dconv, merge_x.shape:", final_x.shape) # [B, d_inner, L]

        x = torch.transpose(final_x, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        x = self.out_norm(x)
        out = self.out_proj(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)

        x = input * self.skip_scale + self.self_attention(x)
        # print("VSSB1, x.shape", x.shape)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # print("VSSB2, x.shape", x.shape)
        return x.permute(0, 3, 1, 2).contiguous()


class MultiMambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.recons = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                norm_layer=nn.LayerNorm,
                d_state=16,
            ))
            self.recons.append(PatchUnembed(
                embed_dim=dim, 
                out_chans=1, 
                upsample_factor=2,
            ))
        self.conv2d = nn.Conv2d(dim+1, dim, 1,1,0) # change channels

    def forward(self, x):
        output_list = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            output = torch.clamp(self.recons[i](x), 0, 1)
            output_list.append(output)
            down_output = F.interpolate(output, scale_factor=0.5, mode='bilinear', align_corners=False)
            x = self.conv2d(torch.cat((down_output, x), dim=1))

        return output_list


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=64, embed_dim=128):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(in_chans)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        return x

class PatchUnembed(nn.Module):
    def __init__(self, embed_dim=64, out_chans=1, upsample_factor=2):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upsample_factor)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim // (upsample_factor**2), embed_dim // (2*(upsample_factor**2)), kernel_size=3, stride=1, padding=1),
            nn.PReLU(embed_dim // (2*(upsample_factor**2))),
            nn.Conv2d(embed_dim // (2*(upsample_factor**2)), out_chans, kernel_size=3, stride=1, padding=1),
            nn.PReLU(out_chans))
        self.norm = nn.LayerNorm(embed_dim // (upsample_factor**2))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        return x
    

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class Spk2ImgMamba(BasicModel):
    def __init__(self,
                 img_size=96,
                 in_chans=32,
                 embed_dim=48,
                 layer_num=3,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super(Spk2ImgMamba, self).__init__()
        
        # Representation
        self.resnet = ResidualBlock(in_channels=61, num_channles=17, use_1x1conv=True)
        # change channel 
        self.link_block = ConvBlock(in_chans,embed_dim) # 20->32
        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(5, 2, embed_dim, embed_dim) # 96*96 -> 48*48
        self.mamba_block = MultiMambaBlock(embed_dim, layer_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B,c,h,w=x.size() # C=61
        # Direct Concatenation
        input = torch.cat((TFP(x, channel_step=2), self.resnet(x)), dim=1) # [B, C_in=32, H, W]

        input = self.patch_embed(self.link_block(input)).cuda()
        output_list = self.mamba_block(input)
        return output_list
