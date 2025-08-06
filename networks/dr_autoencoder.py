import torch
import torch.nn as nn
import torch.nn.functional as F

class ReduceAE(nn.Module):
    
    def __init__(self):
        super(ReduceAE, self).__init__()
        self.unet_encoder1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding = 'same'), 
            nn.ReLU(),
            # nn.Conv3d(32, 32, 3, padding = 'same'),
            # nn.ReLU()
        )
        self.unet_encoder2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding = 'same'), 
            nn.ReLU(),
            # nn.Conv3d(64, 64, 3, padding = 'same'),
            # nn.ReLU()
        )
        self.unet_encoder3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding = 'same'), 
            nn.ReLU(),
            # nn.Conv3d(128, 128, 3, padding = 'same'),
            # nn.ReLU()
        )
        self.unet_encoder4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding = 'same'),
            nn.ReLU(),
            # nn.Conv3d(256, 256, 3, padding = 'same'),
            # nn.ReLU()
        )
        
        self.pool = nn.MaxPool3d((2,1,1))
        
        self.unet_encoder5 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding = 'same'), 
            nn.ReLU(),
            # nn.Conv3d(512, 512, 3, padding = 'same'),
            # nn.ReLU()
        )

        self.conv_transpose1 = nn.ConvTranspose3d(512, 256, (3,1,1), stride = (2,1,1), padding = 0)
        
        
        self.unet_decoder1 = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        
        self.conv_transpose2 = nn.ConvTranspose3d(256, 128, (3,1,1), stride = (2,1,1), padding = 0)
        
        self.unet_decoder2 = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding= 1),
            nn.ReLU(),
            # nn.Conv3d(128, 128, 3, padding= 1),
            # nn.ReLU()
        )
        
        self.conv_transpose3 = nn.ConvTranspose3d(128, 64, (2,1,1), stride = (2,1,1), padding = 0)
        
        self.unet_decoder3 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding= 1),
            nn.ReLU(),
            # nn.Conv3d(64, 64, 3, padding= 1),
            # nn.ReLU()
        )
        
        self.conv_transpose4 = nn.ConvTranspose3d(64, 32, (2,1,1), stride = (2,1,1), padding = 0)
        
        self.unet_decoder4 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding= 1),
            nn.ReLU(),
            # nn.Conv3d(32, 32, 3, padding= 1),
            # nn.ReLU()
        )
        
        self.unet_decoder_final = nn.Sequential(
            nn.Conv3d(32, 1, 1),
            nn.Sigmoid()
#             nn.ReLU()
        )
        
    def forward(self, x):
        x_encoded1 = self.unet_encoder1(x)
        x_pool = self.pool(x_encoded1)
        
        x_encoded2 = self.unet_encoder2(x_pool)
        x_pool = self.pool(x_encoded2)
        
        x_encoded3 = self.unet_encoder3(x_pool)
        # x_pool = self.pool(x_encoded3)
        
        # x_encoded4 = self.unet_encoder4(x_pool)
        # x_pool = self.pool(x_encoded4)
        
        # x_encoded5 = self.unet_encoder5(x_pool)
    
        # x_up1 = self.conv_transpose1(x_encoded5)
        # x_decoded = self.unet_decoder1(torch.cat((x_up1, x_encoded4), dim = 1))
        
        # x_up2 = self.conv_transpose2(x_decoded)
        # x_decoded = self.unet_decoder2(torch.cat((x_up2, x_encoded3), dim = 1))
        
        x_up3 = self.conv_transpose3(x_encoded3)
        x_decoded = self.unet_decoder3(torch.cat((x_up3, x_encoded2), dim = 1))
        
        x_up4 = self.conv_transpose4(x_decoded)
        x_decoded = self.unet_decoder4(torch.cat((x_up4, x_encoded1), dim = 1))
        
        output = self.unet_decoder_final(x_decoded)
                          
        return output
