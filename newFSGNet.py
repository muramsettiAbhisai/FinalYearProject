import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class DropBlock(nn.Module):
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
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = torch.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = torch.sigmoid(self.conv(x))
        return x


class DualAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(DualAttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AtrousSpatialPyramidPooling, self).__init__()
        
        # ASPP with dilated convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global feature
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.size()[2:]
        
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.out_conv(x)
        
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            nn.BatchNorm2d(self.inter_channels),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        psi_f = self.psi(f)

        return psi_f


class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(M_Conv, self).__init__()
        pad_size = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=pad_size, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
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

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        mean_a = self.boxfilter(l_a) / N
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        mean_ay = self.boxfilter(l_a * lr_y) / N
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        mean_ax = self.boxfilter(l_a * lr_x) / N

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
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class ConvNext(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        pad_size = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad_size, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_block = DropBlock(7, 0.5)

    def forward(self, x):
        _input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.gamma.unsqueeze(-1).unsqueeze(-1) * x
        x = _input + self.drop_block(x)

        return x


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class FSGNet(nn.Module):
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(FSGNet, self).__init__()

        # Enhanced encoder path
        self.input_layer = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=kernel_size),
            *[ConvNext(base_c * 1, kernel_size=kernel_size) for _ in range(depths[0])]
        )
        self.input_skip = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=kernel_size),
        )
        self.conv1 = M_Conv(channel, base_c * 1, kernel_size=3)
        
        # Add dual attention mechanism after initial feature extraction
        self.dual_attention1 = DualAttentionBlock(base_c * 2)

        # Enhanced downsampling path with improved feature extraction
        self.down_conv_2 = nn.Sequential(*[
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=2, stride=2),
            *[ConvNext(base_c * 2, kernel_size=kernel_size) for _ in range(depths[1])]
            ])
        self.conv2 = M_Conv(channel, base_c * 2, kernel_size=3)
        self.dual_attention2 = DualAttentionBlock(base_c * 4)

        self.down_conv_3 = nn.Sequential(*[
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=2, stride=2),
            *[ConvNext(base_c * 4, kernel_size=kernel_size) for _ in range(depths[2])]
            ])
        self.conv3 = M_Conv(channel, base_c * 4, kernel_size=3)
        self.dual_attention3 = DualAttentionBlock(base_c * 8)

        self.down_conv_4 = nn.Sequential(*[
            nn.Conv2d(base_c * 8, base_c * 8, kernel_size=2, stride=2),
            *[ConvNext(base_c * 8, kernel_size=kernel_size) for _ in range(depths[3])]
            ])
        
        # Add ASPP module for multi-scale feature extraction
        self.aspp = AtrousSpatialPyramidPooling(base_c * 8, base_c * 8)

        # Enhanced decoder path with improved residual blocks
        self.up_residual_conv3 = ResidualConv(base_c * 8, base_c * 4, 1, 1)
        self.up_residual_conv2 = ResidualConv(base_c * 4, base_c * 2, 1, 1)
        self.up_residual_conv1 = ResidualConv(base_c * 2, base_c * 1, 1, 1)

        # Output layers remain the same to maintain output structure
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

        # Enhanced guided filter for edge-aware feature integration
        self.fgf = FastGuidedFilter_attention(r=2, eps=1e-2)
        
        # Cross-attention blocks for better feature correlation
        self.attention_block3 = CrossAttentionBlock(in_channels=base_c * 8)
        self.attention_block2 = CrossAttentionBlock(in_channels=base_c * 4)
        self.attention_block1 = CrossAttentionBlock(in_channels=base_c * 2)

        # Feature fusion blocks with 1x1 convolutions
        self.conv_cat_3 = M_Conv(base_c * 8 + base_c * 8, base_c * 8, kernel_size=1)
        self.conv_cat_2 = M_Conv(base_c * 8 + base_c * 4, base_c * 4, kernel_size=1)
        self.conv_cat_1 = M_Conv(base_c * 4 + base_c * 2, base_c * 2, kernel_size=1)

    def forward(self, x):
        # Get multi-scale inputs
        _, _, h, w = x.size()

        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        # Enhanced encoder path with attention modules
        x1 = self.input_layer(x) + self.input_skip(x)
        x1_conv = self.conv1(x)
        x1_down = torch.cat([x1_conv, x1], dim=1)
        x1_down = self.dual_attention1(x1_down)  # Apply dual attention

        x2 = self.down_conv_2(x1_down)
        x2_conv = self.conv2(x_scale_2)
        x2_down = torch.cat([x2_conv, x2], dim=1)
        x2_down = self.dual_attention2(x2_down)  # Apply dual attention

        x3 = self.down_conv_3(x2_down)
        x3_conv = self.conv3(x_scale_3)
        x3_down = torch.cat([x3_conv, x3], dim=1)
        x3_down = self.dual_attention3(x3_down)  # Apply dual attention

        x4 = self.down_conv_4(x3_down)
        x4 = self.aspp(x4)  # Apply ASPP for multi-scale features

        # Enhanced decoder path with guided filter and cross-attention
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

        # Generate outputs at multiple scales
        _, _, h, w = x.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        return out_1, out_2, out_3
