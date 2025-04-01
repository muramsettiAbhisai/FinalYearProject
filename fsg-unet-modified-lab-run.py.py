import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


class DropBlock3D(nn.Module):
    def __init__(self, block_size=5, p=0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x):
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            
            # Extend mask across channels in groups to create 3D effect
            c_groups = max(1, C // 4)
            for i in range(c_groups):
                start_c = (C // c_groups) * i
                end_c = (C // c_groups) * (i + 1)
                mask[:, start_c:end_c] = mask[:, start_c:end_c].mean(dim=1, keepdim=True)
                
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):  # Add in_channels parameter
        super(SelfAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),  # Dynamic channels
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),  # Dynamic channels
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial attention
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        x3 = torch.cat((x1, x2), dim=1)
        spatial_attn = torch.sigmoid(self.conv(x3))
        
        # Channel attention
        channel_attn = self.channel_attention(x)
        
        # Combined attention
        x = spatial_attn * channel_attn * x
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            nn.GroupNorm(min(32, self.inter_channels), self.inter_channels),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(min(32, self.inter_channels), self.inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        
        # Add channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=True)
        f = F.gelu(theta_x + phi_g)

        psi_f = self.psi(f)
        
        # Apply channel attention
        c_attn = self.channel_attn(x)
        
        # Combine spatial and channel attention
        return psi_f * c_attn


class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(M_Conv, self).__init__()
        pad_size = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=pad_size, stride=1),
            nn.GroupNorm(min(32, output_channels), output_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        # Feature enhancement before filtering
        lr_x_enhanced = lr_x + torch.sigmoid(lr_x) * 0.1
        lr_y_enhanced = lr_y + torch.sigmoid(lr_y) * 0.1
        
        lr_x_enhanced = lr_x_enhanced.double()
        lr_y_enhanced = lr_y_enhanced.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        N = self.boxfilter(Variable(lr_x_enhanced.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        mean_a = self.boxfilter(l_a) / N
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x_enhanced * lr_y_enhanced) / N
        mean_tax = self.boxfilter(l_t * l_a * lr_x_enhanced) / N
        mean_ay = self.boxfilter(l_a * lr_y_enhanced) / N
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x_enhanced * lr_x_enhanced) / N
        mean_ax = self.boxfilter(l_a * lr_x_enhanced) / N

        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A*hr_x+mean_b).float()


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.GroupNorm(min(32, input_dim), input_dim),
            nn.GELU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.GroupNorm(min(32, output_dim), output_dim),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(min(32, output_dim), output_dim),
        )
        
        # Calculate proper kernel size for ECA
        k_size = int(abs(math.log(output_dim, 2) + 1) / 2)
        k_size = k_size if k_size % 2 else k_size + 1
        
        # Separate the ECA components for more control
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        main_path = self.conv_block(x)
        skip_path = self.conv_skip(x)
        
        # Apply ECA properly
        y = self.avg_pool(main_path)  # [B, C, 1, 1]
        
        # Reshape to [B, 1, C] for 1D convolution
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        
        y = self.conv_eca(y)  # Apply 1D convolution
        y = self.sigmoid(y)  # Apply sigmoid
        
        # Reshape back to [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Apply channel attention
        return main_path * y + skip_path


class LiteViTBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        self.norm1 = LayerNorm(dim, data_format="channels_first")
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.norm2 = LayerNorm(dim, data_format="channels_first")
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_block = DropBlock3D(7, 0.5)

    def forward(self, x):
        _input = x
        x = x + self.drop_block(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_block(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.GELU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.GELU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class FSGNet(nn.Module):
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(FSGNet, self).__init__()

        self.input_layer = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=kernel_size),
            *[LiteViTBlock(base_c * 1, kernel_size=kernel_size) for _ in range(depths[0])]
        )
        self.input_skip = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=kernel_size),
        )
        self.conv1 = M_Conv(channel, base_c * 1, kernel_size=3)

        self.down_conv_2 = nn.Sequential(*[
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=2, stride=2),
            *[LiteViTBlock(base_c * 2, kernel_size=kernel_size) for _ in range(depths[1])]
            ])
        self.conv2 = M_Conv(channel, base_c * 2, kernel_size=3)

        self.down_conv_3 = nn.Sequential(*[
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=2, stride=2),
            *[LiteViTBlock(base_c * 4, kernel_size=kernel_size) for _ in range(depths[2])]
            ])
        self.conv3 = M_Conv(channel, base_c * 4, kernel_size=3)

        # After the self.conv3 line (around line 288)
        self.feature_aggregation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_c * 1, base_c * 1, kernel_size=1),
                nn.GroupNorm(min(32, base_c * 1), base_c * 1),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(base_c * 2, base_c * 2, kernel_size=1),
                nn.GroupNorm(min(32, base_c * 2), base_c * 2),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(base_c * 4, base_c * 4, kernel_size=1),
                nn.GroupNorm(min(32, base_c * 4), base_c * 4),
                nn.GELU()
            )
        ])

        self.down_conv_4 = nn.Sequential(*[
            nn.Conv2d(base_c * 8, base_c * 8, kernel_size=2, stride=2),
            *[LiteViTBlock(base_c * 8, kernel_size=kernel_size) for _ in range(depths[3])]
            ])
        self.attn = SelfAttentionBlock(base_c * 8)

        self.up_residual_conv3 = ResidualConv(base_c * 8, base_c * 4, 1, 1)
        self.up_residual_conv2 = ResidualConv(base_c * 4, base_c * 2, 1, 1)
        self.up_residual_conv1 = ResidualConv(base_c * 2, base_c * 1, 1, 1)

        self.output_layer3 = nn.Sequential(
            nn.Conv2d(base_c * 4, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(base_c * 2, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer1 = nn.Sequential(
            nn.Conv2d(base_c * 1, n_classes, 1, 1),
            nn.Sigmoid(),
        )

        self.fgf = FastGuidedFilter_attention(r=2, eps=1e-2)
        self.attention_block3 = CrossAttentionBlock(in_channels=base_c * 8)
        self.attention_block2 = CrossAttentionBlock(in_channels=base_c * 4)
        self.attention_block1 = CrossAttentionBlock(in_channels=base_c * 2)

        self.conv_cat_3 = M_Conv(base_c * 8 + base_c * 8, base_c * 8, kernel_size=1)
        self.conv_cat_2 = M_Conv(base_c * 8 + base_c * 4, base_c * 4, kernel_size=1)
        self.conv_cat_1 = M_Conv(base_c * 4 + base_c * 2, base_c * 2, kernel_size=1)


        self.boundary_layer3 = nn.Sequential(
            nn.Conv2d(base_c * 4, base_c * 2, 3, padding=1),
            nn.GroupNorm(min(32, base_c * 2), base_c * 2),
            nn.GELU(),
            nn.Conv2d(base_c * 2, 1, 1),
            nn.Sigmoid(),
        )
        self.boundary_layer2 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, 3, padding=1),
            nn.GroupNorm(min(32, base_c), base_c),
            nn.GELU(),
            nn.Conv2d(base_c, 1, 1),
            nn.Sigmoid(),
        )
        self.boundary_layer1 = nn.Sequential(
            nn.Conv2d(base_c * 1, base_c // 2, 3, padding=1),
            nn.GroupNorm(min(32, base_c // 2), base_c // 2),
            nn.GELU(),
            nn.Conv2d(base_c // 2, 1, 1),
            nn.Sigmoid(),
        )

        self.edge_fusion = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        # Get multi-scale from input
        _, _, h, w = x.size()

        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        # Encoder
        x1 = self.input_layer(x) + self.input_skip(x)
        x1_conv = self.conv1(x)
        x1_down = torch.cat([x1_conv, x1], dim=1)

        x2 = self.down_conv_2(x1_down)
        x2_conv = self.conv2(x_scale_2)
        x2_down = torch.cat([x2_conv, x2], dim=1)

        x3 = self.down_conv_3(x2_down)
        x3_conv = self.conv3(x_scale_3)
        x3_down = torch.cat([x3_conv, x3], dim=1)

        x4 = self.down_conv_4(x3_down)
        x4 = self.attn(x4)

        # Decoder
        _, _, h, w = x3_down.size()
        x3_gf = torch.cat([x3_down, F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x3_gf_conv = self.conv_cat_3(x3_gf)
        x3_small = F.interpolate(x3_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x3_small, x4, x3_gf_conv, self.attention_block3(x3_small, x4))
        x3_up = self.up_residual_conv3(fgf_out)

        _, _, h, w = x2_down.size()
        x2_gf = torch.cat([x2_down, F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        x2_small = F.interpolate(x2_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x2_small, x3_up, x2_gf_conv, self.attention_block2(x2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)

        _, _, h, w = x1_down.size()
        x1_gf = torch.cat([x1_down, F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        x1_small = F.interpolate(x1_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x1_small, x2_up, x1_gf_conv, self.attention_block1(x1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)

        _, _, h, w = x.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        # Calculate boundary awareness outputs
        boundary_3 = self.boundary_layer3(x3_up)
        boundary_2 = self.boundary_layer2(x2_up)
        boundary_1 = self.boundary_layer1(x1_up)

        # Resize to match size
        boundary_3 = F.interpolate(boundary_3, size=(h, w), mode='bilinear', align_corners=True)
        boundary_2 = F.interpolate(boundary_2, size=(h, w), mode='bilinear', align_corners=True)

        # Fuse boundary information with segmentation
        boundary_fused = self.edge_fusion(torch.cat([boundary_1, boundary_2, boundary_3], dim=1))
        out_1 = torch.clamp(out_1 * (1 + 0.5 * boundary_fused), 0, 1)
        out_2 = torch.clamp(out_2 * (1 + 0.5 * boundary_fused), 0, 1)
        out_3 = torch.clamp(out_3 * (1 + 0.5 * boundary_fused), 0, 1)

        return out_1, out_2, out_3
