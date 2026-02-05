# -*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import ndjson
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from torchvision import transforms, datasets
from PIL import Image
from torch.autograd import Variable

from model import Discriminator
from model import Generator
from data_loader import DataLoader
from logger import Logger
from utils import make_gif

import re
import torch.nn.functional as F # 자주 이용되는 함수'F'로 설정
import mediapipe as mp  
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import scipy.misc
import numpy as np
import glob
from utils import *
import sys
from argparse import ArgumentParser
from datetime import datetime
from random import shuffle

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["TORCH_USE_CUDA_DSA"] = '0'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__.lower()
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.constant_(m.bias, 0)
    elif classname.find('batchnorm') != -1:
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0)

# Text Logger
def setup_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('training_log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

# Saves [3, 64, 64] tensor x as image.
def save_img(x, filename): 
    x = denorm(x)
    x = x.squeeze()
    to_pil = ToPILImage()
    img = to_pil(x)
    img.save(filename)

def denorm(x):
    out = (x + 1.0) / 2.0
    return out#nn.Tanh(out)

def main():
    if not os.path.exists("./videogan/checkpoints"):
        os.makedirs("./videogan/checkpoints")

    DIR_TO_SAVE = "./videogan/genvideos/"
    if not os.path.exists(DIR_TO_SAVE):
        os.makedirs(DIR_TO_SAVE)
    
    num_epoch = 15
    batch_size_per_gpu = 32
    num_gpus = torch.cuda.device_count()
    total_batch_size = batch_size_per_gpu * num_gpus
    lr = 0.0002
    l1_lambda = 10
    z_dim = 100 # Define this based on Generator input requirement

    device_ids = [0, 1, 2, 3]
    output_device = 0
    device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")

    text_logger = setup_logger('Train')
    logger = Logger('./logs')
    
    # Model Define
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # Load GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        discriminator = nn.DataParallel(discriminator, device_ids=device_ids, output_device=output_device)
        generator = nn.DataParallel(generator, device_ids=device_ids, output_device=output_device)
    
    # Optimizer
    params_G = list(filter(lambda p: p.requires_grad, generator.parameters()))
    g_optim = torch.optim.Adam(params_G, lr=0.0002, betas=(0.5,0.999))
    params_D = list(filter(lambda p: p.requires_grad, discriminator.parameters()))
    d_optim = torch.optim.Adam(params_D, lr=0.0002, betas=(0.5,0.999))

    loss_function = nn.CrossEntropyLoss()

    dataloader = DataLoader(total_batch_size)
    data_size = len(dataloader.train_index)
    num_batch = data_size // total_batch_size # 60000 // 32 = 1875

    text_logger.info('Total number of videos for train = ' + str(data_size))
    text_logger.info('Total number of batches per echo = ' + str(num_batch))

    start_time = time.time()
    counter = 0

    for current_epoch in tqdm(range(1, num_epoch+1)): # 5
        n_updates = 1
        for batch_index in range(num_batch): # 32 data processing per 1875 times
            # Data
            # [Batch, Frame(Time), channel, height, width]
            # (32, 128, 3, 64, 64)
            videos = dataloader.get_batch('train')
            real_video = videos.permute(0,2,1,3,4).to(device) # [32, 3, 128, 64, 64] # [B, C, T, H, W]

            # Training the generator and discriminator alternately
            torch.cuda.empty_cache()
            if n_updates % 2 == 1: # Trainig Discriminator
                # Noise - Pre-generate Gaussian noise for batch 
                noise = torch.from_numpy(np.random.normal(0, 1, size=[total_batch_size, z_dim]).astype(np.float32)).to(device)

                # Initialize grad
                d_optim.zero_grad()

                # Generate Fake Video using Generator and noise
                with torch.no_grad():
                    fake_video, foreground, background, mask = generator(noise)

                # Loss(Fake Video, Fake Logit, Real Video, Real Logit)
                real_logit = discriminator(real_video)
                fake_logit = discriminator(fake_video.detach())

                real_prob = torch.mean(torch.sigmoid(real_logit))
                fake_prob = torch.mean(torch.sigmoid(fake_logit))

                real_loss = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
                fake_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                d_optim.step()
                info = {
                    'd_loss': d_loss.item()
                }
                for tag,value in info.items():
                    logger.scalar_summary(tag, value, counter)
                     
            else: # Trainig Generator
                # Noise - Pre-generate Gaussian noise for batch 
                noise = torch.from_numpy(np.random.normal(0, 1, size=[total_batch_size, z_dim]).astype(np.float32)).to(device)

                # Initialize grad
                g_optim.zero_grad()

                # Generate Fake Video using Generator and noise
                gen_video, foreground, background, mask = generator(noise)

                # Loss(Gen Video, Gen Logit, Real Video, Real Logit)
                gen_logit = discriminator(gen_video)

                gen_loss = F.binary_cross_entropy_with_logits(gen_logit, torch.ones_like(gen_logit))
                g_loss = gen_loss + 0.1 * F.l1_loss(mask, torch.zeros_like(mask), reduction='mean')

                g_loss.backward()
                g_optim.step()
                info = {
                    'g_loss' : g_loss.item()
                }
                for tag,value in info.items():
                    logger.scalar_summary(tag, value, counter)

                '''
                # Calculate validation loss
                videos = to_variable(dataloader.get_batch('test').permute(0,2,1,3,4)).to(device) # [64,3, 32, 64, 64]
                first_frame = videos[:,:,0:1,:,:]
                fake_videos = generator(first_frame)
                outputs = discriminator(fake_videos).squeeze()
                gen_first_frame = fake_videos[:,:,0:1,:,:]
                err = torch.mean(torch.abs(first_frame - gen_first_frame)) * l1_lambda
                g_val_loss = loss_function(outputs, real_labels) + err
                info = {
                'g_val_loss' : g_val_loss.data[0],
                }
                for tag,value in info.items():
                    logger.scalar_summary(tag, value, counter)
                '''
            
            n_updates += 1

            if (batch_index + 1) % 5 == 0:
                text_logger.info("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, \
                                 time: %4.4f" \
                                 % (current_epoch, num_epoch, batch_index+1, num_batch, \
                                 d_loss.item(), g_loss.item(), time.time()-start_time))

            counter += 1

            # intermediate save 
            if (batch_index + 1) % 100 == 0:
                make_gif(denorm(gen_video.data.cpu()[0]), DIR_TO_SAVE + 'fake_gifs_sample__%s_%s_b.gif' % (current_epoch, batch_index))
                #process_and_write_video(gen_video[0:1].cpu().data.numpy(), DIR_TO_SAVE + 'fake_gifs_train__%s_%s_b.gif' % (current_epoch, batch_index))
                text_logger.info('Gifs saved at epoch: %d, batch_index: %d' % (current_epoch, batch_index))

            if (batch_index + 1) % 1000 == 0:
                torch.save(generator.state_dict(), './generator.pkl')
                torch.save(discriminator.state_dict(), './discriminator.pkl')
                text_logger.info('Saved the model to generator.pkl and discriminator.pkl')
            
            # Decay the learning rate
            if (batch_index + 1) % 1000 == 0:
                lr = lr / 10.0
                text_logger.info('Decayed learning rate to %.16f' % lr)
                for param_group in d_optim.param_groups:
                    param_group['lr'] = lr
                for param_group in g_optim.param_groups:
                    param_group['lr'] = lr

        # Save checkpoint after each epoch
        torch.save(generator.state_dict(), f'./videogan/checkpoints/generator_epoch_{current_epoch}.pkl')
        torch.save(discriminator.state_dict(), f'./videogan/checkpoints/discriminator_epoch_{current_epoch}.pkl')
        text_logger.info(f'Models saved for epoch {current_epoch}')

if __name__ == "__main__":
    main()