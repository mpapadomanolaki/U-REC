import torch.nn.functional as F

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        #down1
        self.conv_1_1 = conv_block(n_channels, 32)
        self.conv_1_2 = conv_block(32,32)
        self.maxp_1 = MaxP(32,32)

        #down2
        self.conv_2_1 = conv_block(32, 64)
        self.conv_2_2 = conv_block(64,64)
        self.maxp_2 = MaxP(64,64)

        #down3
        self.conv_3_1 = conv_block(64,128)
        self.conv_3_2 = conv_block(128,128)
        self.conv_3_3 = conv_block(128,128)
        self.maxp_3 = MaxP(128,128)

        #down4
        self.conv_4_1 = conv_block(128,256)
        self.conv_4_2 = conv_block(256,256)
        self.conv_4_3 = conv_block(256,256)
        self.maxp_4 = MaxP(256,256)

        #down5
        self.conv_5_1 = conv_block(256,512)
        self.conv_5_2 = conv_block(512,512)
        self.conv_5_3 = conv_block(512,512)
        self.maxp_5 = MaxP(512,512)

        #up1
        self.D_conv_1 = conv_block(512,512)
        self.D_up_1 = up(512,512)

        #up2
        self.D_conv_2_1 = conv_block(512*2, 512)
        self.D_conv_2_2 = conv_block(512, 512)
        self.D_conv_2_3 = conv_block(512, 256)
        self.D_conv_2_4 = conv_block(256, 256)
        self.D_up_2 = up(256,256)

        #up3
        self.D_conv_3_1 = conv_block(256*2, 256)
        self.D_conv_3_2 = conv_block(256, 256)
        self.D_conv_3_3 = conv_block(256, 128)
        self.D_conv_3_4 = conv_block(128,128)
        self.D_up_3 = up(128,128)

        #up4
        self.D_conv_4_1 = conv_block(128*2,128)
        self.D_conv_4_2 = conv_block(128,128)
        self.D_conv_4_3 = conv_block(128, 64)
        self.D_conv_4_4 = conv_block(64,64)
        self.D_up_4 = up(64,64)

        #up5
        self.D_conv_5_1 = conv_block(64*2,64)
        self.D_conv_5_2 = conv_block(64,32)
        self.D_conv_5_3 = conv_block(32,32)
        self.D_up_5 = up(32,32)

        self.final_block = conv_block(32*2,32)
        self.outconv = outconv(32,n_classes)

    def encode(self, x):
        x = self.conv_1_1(x)
        x1 = self.conv_1_2(x)
        x = self.maxp_1(x1)

        x = self.conv_2_1(x)
        x2 = self.conv_2_2(x)
        x = self.maxp_2(x2)

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x3 = self.conv_3_3(x)
        x = self.maxp_3(x3)

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x4 = self.conv_4_3(x)
        x = self.maxp_4(x4)

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x5 = self.conv_5_3(x)
        x_mp = self.maxp_5(x5)

        return x1,x2,x3,x4,x5,x_mp

    def decode(self, x1,x2,x3,x4,x5,x_mp):

        m = self.D_conv_1(x_mp)
        m = self.D_up_1(m)
        m = torch.cat((m,x5), 1)

        m = self.D_conv_2_1(m)
        m = self.D_conv_2_2(m)
        m = self.D_conv_2_3(m)
        m = self.D_conv_2_4(m)
        m = self.D_up_2(m)
        m = torch.cat((m,x4), 1)

        m = self.D_conv_3_1(m)
        m = self.D_conv_3_2(m)
        m = self.D_conv_3_3(m)
        m = self.D_conv_3_4(m)
        m = self.D_up_3(m)
        m = torch.cat((m,x3), 1)

        m = self.D_conv_4_1(m)
        m = self.D_conv_4_2(m)
        m = self.D_conv_4_3(m)
        m = self.D_conv_4_4(m)
        m = self.D_up_4(m)
        m = torch.cat((m,x2), 1)

        m = self.D_conv_5_1(m)
        m = self.D_conv_5_2(m)
        m = self.D_conv_5_3(m)
        m = self.D_up_5(m)
        m = torch.cat((m,x1), 1)

        m = self.final_block(m)
        m = self.outconv(m)

        return m


    def forward(self, x):

        x1,x2,x3,x4,x5,x_mp=self.encode(x)  ##input1  !!
 
        segm = self.decode(x1,x2,x3,x4,x5,x_mp)

        return segm
