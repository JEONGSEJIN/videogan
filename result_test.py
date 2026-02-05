# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from PIL import Image
from torch.autograd import Variable
import time
import logging
from model import Discriminator
from model import Generator
from data_loader import DataLoader
from logger import Logger
from utils import make_gif
import ndjson
import os
import cv2
import numpy as np #선형대수 관련 함수 이용 가능 모듈
import matplotlib.pyplot as plt#시각화 모듈
import torch #파이토치 기본모듈
import torch.nn as nn #신경망 모델 설계 시 필요한 함수
import torch.nn.functional as F # 자주 이용되는 함수'F'로 설정
from torchvision import transforms, datasets #torchvision모듈 내 transforms, datasets함수 임포트
from torchvision import datasets, transforms
#from torch.utils.data import DataLoader
import gzip
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import re
import cv2
import numpy as np
import mediapipe as mp
from collections import Counter
import pandas as pd  # Excel 저장용
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from model import Generator
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from utils import make_gif
from PIL import Image
import os
from collections import OrderedDict

# 모델 로딩 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    num_epoch = 15
    #batchSize = 16 # GPU 당 배치 사이즈
    batch_size_per_gpu = 8  # GPU 당 배치 사이즈
    num_gpus = torch.cuda.device_count()
    total_batch_size = batch_size_per_gpu * num_gpus  # 전체 배치 사이즈

    DIR_TO_SAVE = "/raid2/jeongsj/gen_videos_test/"
    
    if not os.path.exists(DIR_TO_SAVE):
        os.makedirs(DIR_TO_SAVE)

    generator = Generator().to(device)
    #generator.load_state_dict(torch.load('/raid2/jeongsj/generator.pkl', map_location=device)) #./generator.pkl', map_location=device))
    state_dict = torch.load('/raid2/jeongsj/generator.pkl', map_location=device)
    new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
    generator.load_state_dict(new_state_dict)
    
    
    generator.eval()

    dataloader = DataLoader(total_batch_size)
    data_size = len(dataloader.train_index)
    num_batch = data_size // total_batch_size#batchSize

    for batch_index in range(num_batch):
        val_videos, filenames = dataloader.get_batch('test')
        val_videos = val_videos.permute(0, 2, 1, 3, 4).to(device)
        first_frame = val_videos[:, :, 0:1, :, :] # test dataset의 첫 번째 frame
        fake_videos = generator(first_frame[0])

        if (batch_index + 1) % 100 == 0:
            save_img(first_frame[0].data.cpu(), DIR_TO_SAVE + 'fake_gifs_%s_a.jpg' % (batch_index))
            make_gif(denorm(fake_videos.data.cpu()[0]), DIR_TO_SAVE + 'fake_gifs_%s_b.gif' % (batch_index))



if __name__ == "__main__":
    main()