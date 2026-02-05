from ops import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Discriminator 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # input: [batch, channels, frame size, height, width]
        #        [-1, 3, 32, 64, 64]  
        # input channel: 3, output channel: 64
        self.model = nn.Sequential( # 
                conv3d(3, 64), #[-1, 64, 16, 32, 32] (input channel: 3, output channel: 64)
                lrelu(0.2),

                conv3d(64, 128), #[-1, 128, 8, 16, 16]
                batchNorm3d(128, 1e-3), 
                lrelu(0.2),

                conv3d(128, 256), #[-1, 256, 4, 8, 8]
                batchNorm3d(256, 1e-3),
                lrelu(0.2),

                conv3d(256, 512), #[-1, 512, 2, 4, 4]
                batchNorm3d(512, 1e-3),
                lrelu(0.2),

                conv3d(512, 1024), #[-1, 512, 2, 4, 4]
                batchNorm3d(1024, 1e-3),
                lrelu(0.2),

                conv3d_last(1024, 1)
        )
        
    def forward(self, x, mask=None):
        out = self.model(x)  # [B, 2, 1, 1, 1]
        #print("discriminator x: ", x.size())
        #print("discriminator out: ", out.size())
        return out

# Background
class G_background(nn.Module):
    def __init__(self, z_dim=100):
        super(G_background, self).__init__()
        self.model = nn.Sequential(
                deconv2d_first(z_dim, 512), # [B, 512, 4, 4]  #[-1,512,4,4]
                batchNorm2d(512),
                relu(),

                deconv2d(512,256), # [B, 256, 8, 8]  #[-1,512,4,4]
                batchNorm2d(256),
                relu(),

                deconv2d(256,128),  # [B, 128, 16, 16]
                batchNorm2d(128),
                relu(),

                deconv2d(128,64),  # [B, 64, 64, 64]
                batchNorm2d(64),
                relu(),

                deconv2d(64,3),  # [B, 3, ?, ?]
                nn.Tanh()
                )

    def forward(self, x):
        out = self.model(x)
        return out

# Foreground
class G_foreground(nn.Module): 
    def __init__(self, z_dim=100):
        super(G_foreground, self).__init__()
        self.model = nn.Sequential(
                deconv3d_first(z_dim, 512), #[-1,512,4,4]
                batchNorm3d(512),
                relu(),

                deconv3d(512,256),
                batchNorm3d(256),
                relu(),
                
                deconv3d(256,128),
                batchNorm3d(128),
                relu(),
                
                deconv3d(128,64),
                batchNorm3d(64),
                relu(),

                deconv3d(64,32),
                batchNorm3d(32),
                relu()
                )

    def forward(self,x):
        out = self.model(x)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=100): # x, z):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        # Background Stream
        self.background = G_background(z_dim) # Background + Tanh
        
        # Foreground Stream
        self.foreground = G_foreground(z_dim)  # Foreground Stream 앞 네 단계
        
        self.gen_net = nn.Sequential(deconv3d(32, 3), nn.Tanh()) # Foreground + Tanh
        self.mask_net = nn.Sequential(deconv3d(32, 1), nn.Sigmoid()) # Mask + Sigmoid

        # noise vector를 처리해서 encoder와 결합할 수 있도록 변환
        #self.fc_noise = nn.Sequential(nn.Linear(z_dim, 1024 * 4 *4), nn.ReLU())

    def forward(self, z):
        # Background Stream
        #print("z : ", z.size())
        z_background_input = z.unsqueeze(2).unsqueeze(3)
        #print("z_background_input : ", z_background_input.size())
        background = self.background(z_background_input).unsqueeze(2)
        #print("background : ", background.size())

        # Foreground Stream
        z_foreground_input = z.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        #print("z_foreground_input : ", z_foreground_input.size())
        foreground_stream = self.foreground(z_foreground_input)
        #print("foreground_stream : ", foreground_stream.size())

        # foreground + Tanh
        foreground = self.gen_net(foreground_stream)
        #print("foreground : ", foreground.size())
        
        # mask + Sigmoid
        mask = self.mask_net(foreground_stream)
        #print("mask : ", mask.size())

        # out = m * f + (1 - m) * b
        out = mask * foreground + (1 - mask) * background
        #print("out : ", out.size())

        return out, foreground, background, mask

class G_encode(nn.Module):
    def __init__(self):
        super(G_encode, self).__init__()
        self.model = nn.Sequential(
                conv2d(3,128),
                relu(),
                conv2d(128,256),
                batchNorm4d(256),
                relu(),
                conv2d(256,512),
                batchNorm4d(512),
                relu(),
                conv2d(512,1024),
                batchNorm4d(1024),
                relu(),
                )
    def forward(self,x):
        #print('G_encode Input =', x.size())
        out = self.model(x)
        #print('G_encode Output =', out.size())
        return out
        
'''
if __name__ == '__main__':
    for i in range(1):
        x = Variable(torch.rand([128, 3, 128, 64, 64]).cuda()) # [batch, channel, frame, height, width]
        model = Discriminator().cuda()
        print('Discriminator input', x.size())
        out = model(x).squeeze()
        print('Discriminator out ', out.size()) 

        #x = Variable(torch.rand([128, 3, 1, 64, 64]).cuda())
        z = torch.randn(128, 100).cuda()
        print('Generator input', z.size())
        model = Generator().cuda()
        out = model(z)  
        print('Generator out ', out.size())
        print(type(out.data[0]))
        print(out.data[0].size())

        x = Variable(torch.rand([13,3,64,64])).cuda()
        #x = Variable(torch.rand([13,3,1,64,64]))
        print('Generator input', x.size())
        model = Generator().cuda()
        out = model(x)  
        print('Generator out ', out.size())
'''