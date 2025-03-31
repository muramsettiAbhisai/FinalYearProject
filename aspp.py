class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # Calculate exact channels per branch to ensure we get the right total
        # We have 5 branches, so each should output out_channels // 5 features
        branch_channels = out_channels // 5
        total_concat_channels = branch_channels * 5  # This is what we'll get after concatenation
        
        self.conv1 = nn.Conv2d(in_channels, branch_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(branch_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, branch_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(branch_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(in_channels, branch_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(branch_channels)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(in_channels, branch_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(branch_channels)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fix: Use the actual concatenated channels count as input
        self.conv_out = nn.Conv2d(total_concat_channels, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        size = x.size()[2:]
        
        feat1 = self.relu1(self.bn1(self.conv1(x)))
        feat2 = self.relu2(self.bn2(self.conv2(x)))
        feat3 = self.relu3(self.bn3(self.conv3(x)))
        feat4 = self.relu4(self.bn4(self.conv4(x)))
        
        feat5 = self.global_avg_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=True)
        
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.conv_out(out)
        out = self.bn_out(out)
        out = self.relu_out(out)
        out = self.dropout(out)
        
        return out
