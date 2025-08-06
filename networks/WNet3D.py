import torch
import torch.nn as nn
import torch.nn.functional as F

class WNet3D(nn.Module):
    
    def __init__(self, k):
        super(WNet3D, self).__init__()
        self.k = k
        self.unet1_enc1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding = 'same'), 
#             nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding = 'same'),
#             nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.unet1_enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding = 'same'), 
#             nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding = 'same'),
#             nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.unet1_enc3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding = 'same'), 
#             nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, padding = 'same'),
#             nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.unet1_enc4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding = 'same'),
#             nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding = 'same'),
#             nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        self.pool_unet1 = nn.MaxPool3d((2,2,2))
        self.pool_unet2 = nn.MaxPool3d((1,2,2))
        self.batchnorm512 = nn.BatchNorm3d(512)
        self.batchnorm256 = nn.BatchNorm3d(256)
        self.batchnorm128 = nn.BatchNorm3d(128)
        self.batchnorm64 = nn.BatchNorm3d(64)
        self.batchnorm32 = nn.BatchNorm3d(32)
        
        self.unet1_enc5 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding = 'same'), 
#             nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding = 'same'),
#             nn.BatchNorm3d(512),
            nn.ReLU()
        )

        self.unet1_up1 = nn.ConvTranspose3d(512, 256, (1,2,2), stride = (1,2,2), padding = 0)
        
        
        self.unet1_dec1 = nn.Sequential(
            nn.Conv3d(256, 256, (1,3,3), padding=(0,1,1)),
#             nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, (1,3,3), padding=(0,1,1)),
#             nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        self.unet1_up2 = nn.ConvTranspose3d(256, 128, (1,2,2), stride = (1,2,2), padding = 0)
        
        self.unet1_dec2 = nn.Sequential(
            nn.Conv3d(128, 128, (1,3,3), padding= (0,1,1)),
#             nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, (1,3,3), padding= (0,1,1)),
#             nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        self.unet1_up3 = nn.ConvTranspose3d(128, 64, (1,2,2), stride = (1,2,2), padding = 0)
        
        self.unet1_dec3 = nn.Sequential(
            nn.Conv3d(64, 64, (1,3,3), padding= (0,1,1)),
#             nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, (1,3,3), padding= (0,1,1)),
#             nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        self.unet1_up4 = nn.ConvTranspose3d(64, 32, (1,2,2), stride = (1,2,2), padding = 0)
        
        self.unet1_dec4 = nn.Sequential(
            nn.Conv3d(32, 32, (1,3,3), padding= (0,1,1)),
#             nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding= (0,1,1)),
#             nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
#         self.unet1_dec_out = nn.Sequential(
#             nn.Conv3d(32, 1, 1),
# #             nn.BatchNorm3d(1),
#             nn.Softmax(dim = 2)
#         )
        

#         self.unet2_enc1 = nn.Sequential(
#             nn.Conv3d(1, 32, (1,3,3), padding = 'same'), 
# #             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.Conv3d(32, 32, (1,3,3), padding = 'same'),
# #             nn.BatchNorm3d(32),
#             nn.ReLU()
#         )
        self.unet1_dec_out = nn.Sequential(
            nn.Conv3d(32, 1, 1),
#             nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Flatten(1,2),
            nn.Conv2d(12, self.k, 1),
#             nn.BatchNorm3d(1),
            nn.Softmax(dim = 1)
        )
        

        self.unet2_enc1 = nn.Sequential(
            nn.Conv2d(self.k, 12, 1),
#             nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Unflatten(1, (1, 12)),
            nn.Conv3d(1, 32, 3, padding = 'same'), 
#             nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding = 'same'),
#             nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
        self.unet2_enc2 = nn.Sequential(
            nn.Conv3d(32, 64, (1,3,3), padding = 'same'), 
#             nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, (1,3,3), padding = 'same'),
#             nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.unet2_enc3 = nn.Sequential(
            nn.Conv3d(64, 128, (1,3,3), padding = 'same'), 
#             nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, (1,3,3), padding = 'same'),
#             nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.unet2_enc4 = nn.Sequential(
            nn.Conv3d(128, 256, (1,3,3), padding = 'same'),
#             nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, (1,3,3), padding = 'same'),
#             nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        self.unet2_enc5 = nn.Sequential(
            nn.Conv3d(256, 512, (1,3,3), padding = 'same'), 
#             nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, (1,3,3), padding = 'same'),
#             nn.BatchNorm3d(512),
            nn.ReLU()
        )

        self.unet2_up1 = nn.ConvTranspose3d(512, 256, 2, stride = 2, padding = 0)
        
        
        self.unet2_dec1 = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1),
#             nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
#             nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        self.unet2_up2 = nn.ConvTranspose3d(256, 128, 2, stride = 2, padding = 0)
        
        self.unet2_dec2 = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding= 1),
#             nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, padding= 1),
#             nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        self.unet2_up3 = nn.ConvTranspose3d(128, 64, 2, stride = 2, padding = 0)
        
        self.unet2_dec3 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding= 1),
#             nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding= 1),
#             nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        self.unet2_up4 = nn.ConvTranspose3d(64, 32, 2, stride = 2, padding = 0)
        
        self.unet2_dec4 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding= 1),
#             nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding= 1),
#             nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
        self.unet2_dec_out = nn.Sequential(
            nn.Conv3d(32, 1, 1),
#             nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ## u-net encoder
        x_u1_e1 = self.unet1_enc1(x)
        x_pool = self.pool_unet1(x_u1_e1)
        x_u1_e2 = self.unet1_enc2(x_pool)
        x_pool = self.pool_unet1(x_u1_e2)
        x_u1_e3 = self.unet1_enc3(x_pool)
        x_pool = self.pool_unet1(x_u1_e3)
        x_u1_e4 = self.unet1_enc4(x_pool)
        x_pool = self.pool_unet1(x_u1_e4)

        x_u1_e5 = self.unet1_enc5(x_pool)
    
        x_u1_up1 = self.unet1_up1(x_u1_e5)
#         x_u1_ct1 = self.unet1_up1(x_u1_e5)
#         shape1 = x_u1_ct1.shape
#         shape2 = x_u1_e4.shape
#         pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
#                shape2[0] - shape1[0])
#         x_u1_ct1= F.pad(x_u1_ct1, pad, "constant", 0)
#         x_u1_up1 = torch.cat((x_u1_ct1,x_u1_e4), dim = 1)
        x_u1_up1 = self.batchnorm256(x_u1_up1)
        x_u1_d1 = self.unet1_dec1(x_u1_up1)
        
        x_u1_up2 = self.unet1_up2(x_u1_d1)
#         x_u1_ct2 = self.unet1_up2(x_u1_d1)
#         shape1 = x_u1_ct2.shape
#         shape2 = x_u1_e3.shape
#         pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
#                shape2[0] - shape1[0])
#         x_u1_ct2 = F.pad(x_u1_ct2, pad, "constant", 0)
        
#         x_u1_up2 = torch.cat((x_u1_ct2, x_u1_e3), dim = 1)
        x_u1_up2 = self.batchnorm128(x_u1_up2)
        x_u1_d2 = self.unet1_dec2(x_u1_up2)
 
        x_u1_up3 = self.unet1_up3(x_u1_d2)
#         x_u1_ct3 = self.unet1_up3(x_u1_d2)
#         shape1 = x_u1_ct3.shape
#         shape2 = x_u1_e2.shape
#         pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
#                shape2[0] - shape1[0])
#         x_conv_transp = F.pad(x_u1_ct3, pad, "constant", 0)
#         x_u1_up3 = torch.cat((x_u1_ct3, x_u1_e2), dim = 1)
        x_u1_up3 = self.batchnorm64(x_u1_up3)
        x_u1_d3 = self.unet1_dec3(x_u1_up3)

        x_u1_up4 = self.unet1_up4(x_u1_d3)
#         x_u1_ct4 = self.unet1_up4(x_u1_d3)
#         shape1 = x_u1_ct4.shape
#         shape2 = x_u1_e1.shape
#         pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
#                shape2[0] - shape1[0])
#         x_u1_ct4 = F.pad(x_u1_ct4, pad, "constant", 0)
        
#         x_u1_up4 = torch.cat((x_u1_ct4, x_u1_e1), dim = 1)
        x_u1_up4 = self.batchnorm32(x_u1_up4)
        x_u1_d4 = self.unet1_dec4(x_u1_up4)
        
        unet1_out = self.unet1_dec_out(x_u1_d4)
         
        ## u-net decoder
        
        x_u2_e1 = self.unet2_enc1(unet1_out)
        x_pool = self.pool_unet2(x_u2_e1)
        x_u2_e2 = self.unet2_enc2(x_pool)
        x_pool = self.pool_unet2(x_u2_e2)
        x_u2_e3 = self.unet2_enc3(x_pool)
        x_pool = self.pool_unet2(x_u2_e3)
        x_u2_e4 = self.unet2_enc4(x_pool)
        x_pool = self.pool_unet2(x_u2_e4)

        x_u2_e5 = self.unet2_enc5(x_pool)
    
        x_u2_ct1 = self.unet2_up1(x_u2_e5)
        shape1 = x_u2_ct1.shape
        shape2 = x_u1_e4.shape
        pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
               shape2[0] - shape1[0])
        x_u2_ct1= F.pad(x_u2_ct1, pad, "constant", 0)
        x_u2_up1 = torch.cat((x_u2_ct1,x_u1_e4), dim = 1)
        x_u2_up1 = self.batchnorm512(x_u2_up1)
        x_u2_d1 = self.unet2_dec1(x_u2_up1)
        
        x_u2_ct2 = self.unet2_up2(x_u2_d1)
        shape1 = x_u2_ct2.shape
        shape2 = x_u1_e3.shape
        pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
               shape2[0] - shape1[0])
        x_u2_ct2 = F.pad(x_u2_ct2, pad, "constant", 0)
        
        x_u2_up2 = torch.cat((x_u2_ct2, x_u1_e3), dim = 1)
        x_u2_up2 = self.batchnorm256(x_u2_up2)
        x_u2_d2 = self.unet2_dec2(x_u2_up2)
 
        
        x_u2_ct3 = self.unet2_up3(x_u2_d2)
        shape1 = x_u2_ct3.shape
        shape2 = x_u1_e2.shape
        pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
               shape2[0] - shape1[0])
        x_u2_ct3 = F.pad(x_u2_ct3, pad, "constant", 0)
        x_u2_up3 = torch.cat((x_u2_ct3, x_u1_e2), dim = 1)
        x_u2_up3 = self.batchnorm128(x_u2_up3)
        x_u2_d3 = self.unet2_dec3(x_u2_up3)

        x_u2_ct4 = self.unet2_up4(x_u2_d3)
        shape1 = x_u2_ct4.shape
        shape2 = x_u1_e1.shape
        pad = (0, shape2[4] - shape1[4], 0, shape2[3] - shape1[3], 0, shape2[2] - shape1[2], 0, 0, 0,
               shape2[0] - shape1[0])
        x_u2_ct4 = F.pad(x_u2_ct4, pad, "constant", 0)
        
        x_u2_up4 = torch.cat((x_u2_ct4, x_u1_e1), dim = 1)
        x_u2_up4 = self.batchnorm64(x_u2_up4)
        x_u2_d4 = self.unet2_dec4(x_u2_up4)
        
        unet2_out = self.unet2_dec_out(x_u2_d4)
        return unet1_out, unet2_out
