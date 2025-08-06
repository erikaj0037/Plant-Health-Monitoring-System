import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3DNorm(nn.Module):
    
    def __init__(self):
        super(UNet3DNorm, self).__init__()
        self.unet_encoder1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding = 'same'), 
#             nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding = 'same'),
#             nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.unet_encoder2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding = 'same'), 
#             nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding = 'same'),
#             nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.unet_encoder3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding = 'same'), 
#             nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, padding = 'same'),
#             nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.unet_encoder4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding = 'same'),
#             nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding = 'same'),
#             nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool3d((2,2,2))
        self.batchnorm512 = nn.BatchNorm3d(512)
        self.batchnorm256 = nn.BatchNorm3d(256)
        self.batchnorm128 = nn.BatchNorm3d(128)
        self.batchnorm64 = nn.BatchNorm3d(64)
        self.batchnorm32 = nn.BatchNorm3d(32)
        
        self.unet_encoder5 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding = 'same'), 
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding = 'same'),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )

        self.conv_transpose1 = nn.ConvTranspose3d(512, 256, 2, stride = 2, padding = 0)
        
        
        self.unet_decoder1 = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1),
#             nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        self.conv_transpose2 = nn.ConvTranspose3d(256, 128, 2, stride = 2, padding = 0)
        
        self.unet_decoder2 = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding= 1),
#             nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, padding= 1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        self.conv_transpose3 = nn.ConvTranspose3d(128, 64, 2, stride = 2, padding = 0)
        
        self.unet_decoder3 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding= 1),
#             nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding= 1),
#             nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        self.conv_transpose4 = nn.ConvTranspose3d(64, 32, 2, stride = 2, padding = 0)
        
        self.unet_decoder4 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding= 1),
#             nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding= 1),
#             nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
        self.unet_decoder_final = nn.Sequential(
            nn.Conv3d(32, 1, 1),
#             nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_encoded1 = self.unet_encoder1(x)
        x_pool = self.pool(x_encoded1)
        x_encoded2 = self.unet_encoder2(x_pool)
        x_pool = self.pool(x_encoded2)
        x_encoded3 = self.unet_encoder3(x_pool)
        x_pool = self.pool(x_encoded3)
        x_encoded4 = self.unet_encoder4(x_pool)
        x_pool = self.pool(x_encoded4)

        x_encoded5 = self.unet_encoder5(x_pool)
    
        x_conv_transp = self.conv_transpose1(x_encoded5)
        shape1 = x_conv_transp.shape
        shape2 = x_encoded4.shape
        pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
               shape2[0] - shape1[0])
        x_conv_transp = F.pad(x_conv_transp, pad, "constant", 0)
        x_up1 = torch.cat((x_conv_transp,x_encoded4), dim = 1)
        x_up1 = self.batchnorm512(x_up1)
        x_decoded = self.unet_decoder1(x_up1)
        
        x_conv_transp = self.conv_transpose2(x_decoded)
        shape1 = x_conv_transp.shape
        shape2 = x_encoded3.shape
        pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
               shape2[0] - shape1[0])
        x_conv_transp = F.pad(x_conv_transp, pad, "constant", 0)
        
        x_up2 = torch.cat((x_conv_transp, x_encoded3), dim = 1)
        x_up2 = self.batchnorm256(x_up2)
        x_decoded = self.unet_decoder2(x_up2)
 
        
        x_conv_transp = self.conv_transpose3(x_decoded)
        shape1 = x_conv_transp.shape
        shape2 = x_encoded2.shape
        pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
               shape2[0] - shape1[0])
        x_conv_transp = F.pad(x_conv_transp, pad, "constant", 0)
        x_up3 = torch.cat((x_conv_transp, x_encoded2), dim = 1)
        x_up3 = self.batchnorm128(x_up3)
        x_decoded = self.unet_decoder3(x_up3)

        x_conv_transp = self.conv_transpose4(x_decoded)
        shape1 = x_conv_transp.shape
        shape2 = x_encoded1.shape
        pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
               shape2[0] - shape1[0])
        x_conv_transp = F.pad(x_conv_transp, pad, "constant", 0)
        
        x_up4 = torch.cat((x_conv_transp, x_encoded1), dim = 1)
        x_up4 = self.batchnorm64(x_up4)
        x_decoded = self.unet_decoder4(x_up4)
        
        output = self.unet_decoder_final(x_decoded)
                          
        return output
