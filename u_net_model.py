import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.max_pool = nn.MaxPool2d(2, 2)
    def forward(self, X):
        return self.down_layer(X)
    
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.down_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.down_layer(X)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # eg in_channels = 1024, out_channels = 512
        # 1024 in_channels from prev layer
        '''
        In ConvTrans : 1024 --> 512
        through skip_connection's 512 : 512 + 512 = 1024 (which is equal to in_channels)
        So input of DoubleConv is also in_channels
        In DoubleConv:
            Conv2D: 1024 --> 512
            Conv2D:  512 --> 512
        
        In ConvTrans : N --> N/2
        through skip_connection's N/2 : N/2 + N/2 = N (which is equal to in_channels)
        So input of DoubleConv is also in_channels
        In DoubleConv:
            Conv2D: N   --> N/2
            Conv2D: N/2 --> N/2
        for next layer, it will go from N/2 --> N/4
        '''
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # 1024 --> 512
        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),  # 1024 --> 512
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False), # 1024 --> 512
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, X, skip_connection):
        X1 = self.up_conv(X)
        if(X1.shape != skip_connection.shape):
            X1 = TF.resize(X1, skip_connection.shape[2:]) # X1 height and width might not remain still same if max_pooling floors the dimension, so match it with the skip_connection height and width
                
        X2 = torch.cat((X1, skip_connection), dim=1) # concatenate skip_connection along channel dimension
        
        return self.DoubleConv(X2)

class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.finalConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # kernel size = 1, since the height and width of the final layer and this should be same(given in paper)
        
    def forward(self, X):
        return self.finalConv(X)
    
class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1):
        super(UNet, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        self.bottleNeck = Down(512, 1024) 
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        self.finalConv = FinalConv(64, out_channels)
        
        
    def forward(self, X):
        
        ### DownSampling
        x1_skip = self.down1(X)          # 003-->064
        x1 = self.max_pool(x1_skip)
        
        x2_skip = self.down2(x1)         # 064->128
        x2 = self.max_pool(x2_skip)
        
        x3_skip = self.down3(x2)         # 128-->256
        x3 = self.max_pool(x3_skip)
        
        x4_skip = self.down4(x3)         # 256-->512
        x4 = self.max_pool(x4_skip)
        
        
        ### BottleNeck Layer
        x5 = self.bottleNeck(x4)         # 512-->1024
        
        
        ### UpSampling        
        x  = self.up1(x5, x4_skip)       # [x5(1024, up_conv will take it to 512) + x4(512)] --> 512
        
        x  = self.up2(x , x3_skip)       # [x ( 512, up_conv will take it to 256) + x3(256)] --> 256
        
        x  = self.up3(x , x2_skip)       # [x ( 256, up_conv will take it to 128) + x2(128)] --> 128
        
        x  = self.up4(x , x1_skip)       # [x ( 128, up_conv will take it to  64) + x1( 64)] --> 64
        
        x  = self.finalConv(x)           # 64 --> 2
        
        return x


def test():
    x = torch.randn(3, 1, 572, 572)
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(f"Preds shape: {preds.shape}")
    print(x.shape == preds.shape)

if __name__ == "__main__":
    test()