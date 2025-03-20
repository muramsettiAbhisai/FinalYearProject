class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, mid_channels=256):
        super(ASPP, self).__init__()
        
        # Branch 1: 1x1 convolution (unchanged)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # New atrous branch with dilation rate 2
        self.branch_atrous1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Updated atrous branch (originally dilation 6, now 3)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Updated atrous branch (originally dilation 12, now 6)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Updated atrous branch (originally dilation 18, now 12)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 5: Global average pooling (unchanged)
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution adjusted for 6 branches
        self.conv_final = nn.Sequential(
            nn.Conv2d(6 * mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        
        h1 = self.branch1(x)          # 1x1 conv
        h_atrous1 = self.branch_atrous1(x)  # Dilation 2
        h2 = self.branch2(x)          # Dilation 3
        h3 = self.branch3(x)          # Dilation 6
        h4 = self.branch4(x)          # Dilation 12
        h5 = F.interpolate(self.branch5(x), size=size, mode='bilinear', align_corners=True)  # Global pooling
        
        # Concatenate all 6 branches
        out = torch.cat([h1, h_atrous1, h2, h3, h4, h5], dim=1)
        out = self.conv_final(out)
        return out
