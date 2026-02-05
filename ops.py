import torch.nn as nn

# Discriminator 꺼 
def conv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def lrelu(negative_slope = 0.2, inplace = True):
    return nn.LeakyReLU(negative_slope, inplace)

def batchNorm3d(num_features, eps = 1e-5): #input: N, C, D, H, W
    return nn.BatchNorm3d(num_features, eps = eps)

def conv3d_last(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(4, 2, 2), stride=(2, 2, 2), padding=0)

# Generator - Foreground (3D) 꺼
def deconv3d_first(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(4,2,2), stride=(1,1,1))

def deconv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

# Generator - Background (2D) 꺼
def deconv2d_first(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = (4,4), stride=(1, 1))

def batchNorm2d(num_features, eps = 1e-5): #input: N, C, H, W
    return nn.BatchNorm2d(num_features, eps = eps)

def relu(inplace = True):
    return nn.ReLU(inplace)

def deconv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

# G_encode
def conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def deconv3d_video(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,1,1))
