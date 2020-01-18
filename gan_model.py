import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import os
from tensorboardX import SummaryWriter

class CS_Dataset(torchvision.datasets.Cityscapes):
    def __init__(self,  root_folder='/cluster/scratch/oezyurty/cityscapes_data', split='train', mode='fine', target_type='semantic', transform=None, target_transform=None, transforms=None):
        super(CS_Dataset, self).__init__(root_folder,split=split,mode=mode,target_type=target_type,transform=transform,target_transform=target_transform)
        self.height = 1024
        self.width = 2048
        self.interp = Image.ANTIALIAS
        self.resize = {}
        self.num_scales = 3

        for ii in range(self.num_scales):
            s = 4 * (2**ii)
            self.resize[ii] = torchvision.transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def __getitem__(self, index):
        CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
        CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]
        inputs = {}
        loaded_img, loaded_sgmn = super(CS_Dataset, self).__getitem__(index)

        for ii in range(self.num_scales):
            inputs[("img", ii)] = self.resize[ii](loaded_img)
            inputs[("segm", ii)] = self.resize[ii](loaded_sgmn)
        
        
        inputs[("cropped")] = torchvision.transforms.Normalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD)(torchvision.transforms.ToTensor()(torchvision.transforms.CenterCrop((256,256))(inputs[("img", 0)])))
        inputs[("cropped_segm")] = torchvision.transforms.ToTensor()(torchvision.transforms.CenterCrop((256,256))(inputs[("segm", 0)]))

        for iii in range(self.num_scales):
            inputs[("img", iii)] = torchvision.transforms.Normalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD)(torchvision.transforms.ToTensor()(inputs[("img", iii)] ))
            inputs[("segm", iii)] = torchvision.transforms.ToTensor()(inputs[("segm", iii)])

            inputs[("segm", iii)] = torch.squeeze(torch.nn.functional.one_hot((torch.round(inputs[("segm", iii)]*255/42)).to(torch.int64), 7).permute(0,3,1,2)).float()

        inputs[("cropped_segm")] = torch.squeeze(torch.nn.functional.one_hot((torch.round(inputs[("cropped_segm")]*255/42)).to(torch.int64), 7).permute(0,3,1,2)).float()

        return inputs


def conv3x3(in_channels, out_channels, dilation_factor=1,stride=1, groups=1, ):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation_factor,bias=False, padding=dilation_factor) #padding=dilation

class ResidualBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, dilation_factor=2, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, dilation_factor=dilation_factor)
        self.norm_layer=nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        #print(identity.shape)
        out = self.conv1(x)
        #print(out.shape)
        out = self.norm_layer(out)
        #print(out.shape)
        out += identity
      
        out = self.relu(out)
        return out
      
class DownBlock(nn.Module):
    def __init__(self, in_feat=3, out_feat=32,kernel_size=3):
        super(DownBlock, self).__init__()

        def down_block(in_feat=3, out_feat=32,kernel_size=3):
            layers = [nn.Conv2d(in_feat, out_feat,kernel_size, stride=2, padding=(kernel_size-1)//2)]
            layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.block = down_block(in_feat=in_feat, out_feat=out_feat,kernel_size=kernel_size)

    def forward(self, x):
        out = self.block(x)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_feat=64, out_feat=32, scale_factor = 2, kernel_size=3, normalize=True,padding=0):
        super(UpBlock, self).__init__()

        def up_block( in_feat, out_feat, scale_factor = 2, kernel_size=3, normalize=True,padding=0):
            layers = [nn.Upsample(scale_factor = scale_factor, mode='bilinear')]
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=1, padding=padding))
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
          
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.block = up_block(in_feat=in_feat, out_feat=out_feat, scale_factor = scale_factor, kernel_size=kernel_size, normalize=normalize,padding=padding)

    def forward(self, x):
        out = self.block(x)
        return out


class ExPGenerator(nn.Module):
    def __init__(self, seg_class_num = 7, seg_encoding_num=7, im_size=(256, 512), exp_size=(256,128)):
        super(ExPGenerator, self).__init__()

            #original image processing part
        self.im_down1 = DownBlock(3,32, kernel_size=7)
        self.im_down2 = DownBlock(32,64, kernel_size=3)
        self.im_down3 = DownBlock(64,128, kernel_size=3)
        self.im_down4 = DownBlock(128,256, kernel_size=3)
            
            #segmentation processing part
        self.seg_down0 = nn.Conv2d(seg_class_num, seg_encoding_num, kernel_size=1, stride=1)
        self.seg_down1 = DownBlock(seg_encoding_num, 32, kernel_size=7)
        self.seg_down2 = DownBlock(32,64, kernel_size=3)
        self.seg_down3 = DownBlock(64,128, kernel_size=3)
        self.seg_down4 = DownBlock(128,256, kernel_size=3)

            #Residual part        
        self.res1 = ResidualBlock(512, 512, dilation_factor=2)
        self.res2 = ResidualBlock(512, 512, dilation_factor=2)
        self.res3 = ResidualBlock(512, 512, dilation_factor=4)
        self.res4 = ResidualBlock(512, 512, dilation_factor=4)
        self.res5 = ResidualBlock(512, 512, dilation_factor=8)
        self.res6 = ResidualBlock(512, 512, dilation_factor=8)

            #Image Upsample for Left Extrapolation
        self.im_up1_left = UpBlock(512,256, kernel_size=3)
        self.im_up2_left = UpBlock(256,128, scale_factor=(1.25 , 1), kernel_size=3)
        self.im_up3_left = UpBlock(128,64, kernel_size=3)
        self.im_up4_left = UpBlock(64,32, scale_factor=(1.25 , 1), kernel_size=3)
        self.im_up5_left = UpBlock(32,16, kernel_size=3)
        self.im_up6_left = UpBlock(16,3, scale_factor=(1.25 , 1), kernel_size=3)
        self.im_up7_left = nn.Upsample(size = exp_size, mode='bilinear')


        #Image Upsample for Right Extrapolation
        self.im_up1_right = UpBlock(512,256, kernel_size=3)
        self.im_up2_right = UpBlock(256,128, scale_factor=(1.25 , 1), kernel_size=3)
        self.im_up3_right = UpBlock(128,64, kernel_size=3)
        self.im_up4_right = UpBlock(64,32, scale_factor=(1.25 , 1), kernel_size=3)
        self.im_up5_right = UpBlock(32,16, kernel_size=3)
        self.im_up6_right = UpBlock(16,3, scale_factor=(1.25 , 1), kernel_size=3)
        self.im_up7_right = nn.Upsample(size = exp_size, mode='bilinear')
      
        #Segmentaion upsample part
        self.seg_up1 = UpBlock(512,256, kernel_size=3)
        self.seg_up2 = UpBlock(256,256, scale_factor=(1 , 1.25), kernel_size=3)
        self.seg_up3 = UpBlock(256,128, kernel_size=3)
        self.seg_up4 = UpBlock(128,128, scale_factor=(1 , 1.25), kernel_size=3)
        self.seg_up5 = UpBlock(128,64, kernel_size=3)
        self.seg_up6 = UpBlock(64,32, scale_factor=(1 , 1.25), kernel_size=3)
        self.seg_up7 = UpBlock(32,3, kernel_size=7, padding=2, normalize=False)
        self.seg_up8 = nn.Upsample(size = im_size, mode='bilinear')
        self.seg_up9 = nn.ConvTranspose2d(3, seg_class_num, kernel_size=1)
        self.seg_out = nn.Softmax(dim=1)
          #  nn.Linear(1024, int(np.prod(img_shape))),  # DUZELT
          #  nn.Tanh()                   # DUZELT
            
        
    def forward(self,  im_in, seg_in): #input_tensor
        '''
        This function produces 3 outputs: 
        im_out_left -> only the left part of the extrapolation with size (3,256,128)
        im_out_right -> only the right part of the extrapolation with size (3,256,128)
        seg_out -> full reconstruction of the segmentation with size (3,256,512)
        '''

        #print(im_in.shape)
        #print(seg_in.shape)
        im_d1 = self.im_down1(im_in)
        im_d2 = self.im_down2(im_d1)
        im_d3 = self.im_down3(im_d2)
        im_d4 = self.im_down4(im_d3)
        
        #print('#'*10)
        #print(type(seg_in))
        #print(seg_in.dtype)
        seg_d0 = self.seg_down0(seg_in)
        seg_d1 = self.seg_down1(seg_d0)
        seg_d2 = self.seg_down2(seg_d1)
        seg_d3 = self.seg_down3(seg_d2)
        seg_d4 = self.seg_down4(seg_d3)
        
        #print('lol')
        #print(im_d4.shape)
        #print(seg_d4.shape)

        conc_l=  torch.cat([im_d4, seg_d4], dim=1)
        #print(conc_l.shape)
        res_l1 = self.res1(conc_l)
        res_l2 = self.res2(res_l1)
        res_l3 = self.res3(res_l2)
        res_l4 = self.res4(res_l3)
        res_l5 = self.res5(res_l4)
        res_l6 = self.res6(res_l5)
        
        im_u1_left = self.im_up1_left(res_l6)
        im_u2_left = self.im_up2_left(im_u1_left)
        im_u3_left = self.im_up3_left(im_u2_left)
        im_u4_left = self.im_up4_left(im_u3_left)
        im_u5_left = self.im_up5_left(im_u4_left)
        im_u6_left = self.im_up6_left(im_u5_left)
        im_out_left = self.im_up7_left(im_u6_left)

        im_u1_right = self.im_up1_right(res_l6)
        im_u2_right = self.im_up2_right(im_u1_right)
        im_u3_right = self.im_up3_right(im_u2_right)
        im_u4_right = self.im_up4_right(im_u3_right)
        im_u5_right = self.im_up5_right(im_u4_right)
        im_u6_right = self.im_up6_right(im_u5_right)
        im_out_right = self.im_up7_right(im_u6_right)
        
        seg_u1 = self.seg_up1(res_l6)
        seg_u2 = self.seg_up2(seg_u1)
        seg_u3 = self.seg_up3(seg_u2)
        seg_u4= self.seg_up4(seg_u3)
        seg_u5= self.seg_up5(seg_u4)
        seg_u6= self.seg_up6(seg_u5)
        seg_u7= self.seg_up7(seg_u6)
        seg_u8= self.seg_up8(seg_u7)
        seg_out_bf_softmax = self.seg_up9(seg_u8)
        seg_out = self.seg_out(seg_out_bf_softmax)

        return im_out_left , im_out_right, seg_out

#Discriminator
def calculate_hw_conv(h_in, w_in, kernel_size, stride=1 , padding=0, dilation=1):
    h = math.floor((h_in + 2*padding - dilation*(kernel_size-1 ) -1)/stride + 1)
    w = math.floor((w_in + 2*padding - dilation*(kernel_size-1 ) -1)/stride + 1) 
    return h , w

class LeftDiscriminator(nn.Module):
    def __init__(self, img_shape = (3, 256, 384)):
        super(LeftDiscriminator, self).__init__()

      #  self.model = nn.Sequential(
      #      nn.Linear(int(np.prod(img_shape)), 512),
      #      nn.LeakyReLU(0.2, inplace=True),
      #      nn.Linear(512, 256),
      #      nn.LeakyReLU(0.2, inplace=True),
      #      nn.Linear(256, 1),
      #      nn.Sigmoid()
      #  )
        ch_num =64
        sz = np.array(img_shape) #np.shape idi
        h, w = calculate_hw_conv(sz[1], sz[2], kernel_size=7, stride=1 , padding=0, dilation=1)
        h, w = calculate_hw_conv(h, w, kernel_size=5, stride=2 , padding=0, dilation=1)
        h, w = calculate_hw_conv(h, w, kernel_size=5, stride=2 , padding=0, dilation=1)
        h, w = calculate_hw_conv(h, w, kernel_size=3, stride=1, padding=0, dilation=1)
        vec_len = int(h*w*ch_num)
      
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, ch_num, 5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(3), 
            nn.Flatten(),
            nn.Linear(38400, 256), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        validity = self.model(img)
        return validity
      
class RightDiscriminator(nn.Module):
    def __init__(self, img_shape = (3, 256, 384)):
        super(RightDiscriminator, self).__init__()
        ch_num =64
        sz = np.array(img_shape) #np.shape idi
        h, w = calculate_hw_conv(sz[1], sz[2], kernel_size=7, stride=1 , padding=0, dilation=1)
        h, w = calculate_hw_conv(h, w, kernel_size=5, stride=2 , padding=0, dilation=1)
        h, w = calculate_hw_conv(h, w, kernel_size=5, stride=2 , padding=0, dilation=1)
        h, w = calculate_hw_conv(h, w, kernel_size=3, stride=1, padding=0, dilation=1)
        vec_len = int(h*w*ch_num)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, ch_num, 5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(3), 
            nn.Flatten(),
            nn.Linear(38400, 256), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        validity = self.model(img)
        return validity

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, scale_factor=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
          
        self.scale_factor = scale_factor
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.Sigmoid()
        )

    def forward(self, img_Full):
        img_input = self.downscale(img_Full)
        return self.model(img_input)
  
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

        