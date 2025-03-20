#!/usr/bin/env python
# -- coding:utf-8 --

import torch
import torch.nn as nn
import torch.nn.functional as F

# Existing utility classes from your code
class ConvBNReLU(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride=1,
                 padding=1,
                 activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in,
                              c_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x

# ASPP Module Definition
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, mid_channels=256):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=18, dilation=18),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(5 * mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h3 = self.branch3(x)
        h4 = self.branch4(x)
        h5 = F.interpolate(self.branch5(x), size=size, mode='bilinear', align_corners=True)
        out = torch.cat([h1, h2, h3, h4, h5], dim=1)
        out = self.conv_final(out)
        return out

# Modified MTUNet with ASPP in the bottleneck
class MTUNet(nn.Module):
    def __init__(self, out_ch=4):
        super(MTUNet, self).__init__()
        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with ASPP
        self.bottleneck = ASPP(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512)  # 512 (upsampled) + 512 (skip)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)   # 256 + 256
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)   # 128 + 128
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)    # 64 + 64
        
        # Output layer
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # Handle single-channel input by repeating to 3 channels
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Encoder
        e1 = self.enc1(x)      # (B, 64, 224, 224)
        p1 = self.pool1(e1)    # (B, 64, 112, 112)
        e2 = self.enc2(p1)     # (B, 128, 112, 112)
        p2 = self.pool2(e2)    # (B, 128, 56, 56)
        e3 = self.enc3(p2)     # (B, 256, 56, 56)
        p3 = self.pool3(e3)    # (B, 256, 28, 28)
        e4 = self.enc4(p3)     # (B, 512, 28, 28)
        p4 = self.pool4(e4)    # (B, 512, 14, 14)
        
        # Bottleneck with ASPP
        b = self.bottleneck(p4)  # (B, 1024, 14, 14)
        
        # Decoder with skip connections
        u1 = self.up1(b)       # (B, 512, 28, 28)
        c1 = torch.cat((u1, e4), dim=1)  # (B, 1024, 28, 28)
        d1 = self.dec1(c1)     # (B, 512, 28, 28)
        
        u2 = self.up2(d1)      # (B, 256, 56, 56)
        c2 = torch.cat((u2, e3), dim=1)  # (B, 512, 56, 56)
        d2 = self.dec2(c2)     # (B, 256, 56, 56)
        
        u3 = self.up3(d2)      # (B, 128, 112, 112)
        c3 = torch.cat((u3, e2), dim=1)  # (B, 256, 112, 112)
        d3 = self.dec3(c3)     # (B, 128, 112, 112)
        
        u4 = self.up4(d3)      # (B, 64, 224, 224)
        c4 = torch.cat((u4, e1), dim=1)  # (B, 128, 224, 224)
        d4 = self.dec4(c4)     # (B, 64, 224, 224)
        
        # Output
        out = self.out(d4)     # (B, out_ch, 224, 224)
        return out

# Example usage (optional, for testing)
if __name__ == "__main__":
    model = MTUNet(out_ch=4)
    x = torch.randn(1, 3, 224, 224)  # Example input
    y = model(x)
    print(y.shape)  # Should output: torch.Size([1, 4, 224, 224])
