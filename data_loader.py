import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import os
import cv2 
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import re
import torchvision.transforms.functional as F
from PIL import Image

# Golf video dataset stabilized by SIFT + RANSAC.
# Downloaded from http://data.csail.mit.edu/videogan/golf.tar.bz2
# Extracting this gives you frames-stable-many folder.
# Also, download the file listings from here: http://data.csail.mit.edu/videogan/golf.txt

#GOLF_DATA_LISTING = '/srv/bat/data/frames-stable-many/golf.txt'
#DATA_ROOT = '/srv/bat/data/frames-stable-many/'

class DataLoader(object):
    def __init__(self, batch_size = 5): 
        #reading data list
        self.batch_size = batch_size # number of batch
        self.crop_size = 64          # image resize (interpolation)
        self.frame_size = 128        # number of frames
        self.image_size = 128        # height, width of frame
        self.train = None
        self.test = None
    
        # reading mnist image
        self.transform = transforms.ToTensor()
        self.mnist_image_train = datasets.MNIST(
            root='./dataset/Image/',
            train=True, 
            download=True, 
            transform=self.transform
        )
        self.mnist_image_test = datasets.MNIST(
            root='./dataset/Image/',
            train=False, 
            download=True, 
            transform=self.transform
        )

        # reading mnist video
        self.mnist_video_train_dir = "./dataset/Video/MNIST/train"
        self.mnist_video_test_dir = "./dataset/Video/MNIST/test"
        self.mnist_video_train = [f for f in os.listdir(self.mnist_video_train_dir) if os.path.isfile(os.path.join(self.mnist_video_train_dir, f))]
        self.mnist_video_test = [f for f in os.listdir(self.mnist_video_test_dir) if os.path.isfile(os.path.join(self.mnist_video_test_dir, f))]

        # Shuffle video index.
        #data_list_path = os.path.join(GOLF_DATA_LISTING) #603776 video path
        #with open(data_list_path, 'r') as f:
        #    self.video_index = [x.strip() for x in f.readlines()]
        #    np.random.shuffle(self.video_index)

        #self.size = len(self.video_index)
        self.train_index = self.mnist_video_train #self.video_index[:self.size//2]
        self.test_index = self.mnist_video_test #self.video_index[self.size//2:]

		# A pointer in the dataset
        self.cursor = 0

    def get_batch(self, type_dataset='train'):
        if type_dataset not in('train', 'test'):
            print(f"type_dataset = {type_dataset} is invalid. Returning None")  # Python 3.6+ print 'type_dataset = ', type_dataset, ' is invalid. Returning None'
            return None
        
        dataset_index = self.train_index if type_dataset == 'train' else self.test_index
        dir_path = self.mnist_video_train_dir if type_dataset == 'train' else self.mnist_video_test_dir

        if self.cursor + self.batch_size > len(dataset_index):
            self.cursor = 0
            np.random.shuffle(dataset_index)

        # [batch size, frame size, channels, height, width]
        t_out = torch.zeros((self.batch_size, self.frame_size, 3, self.crop_size, self.crop_size))
        
        to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
        batch_video_tensor_list = []
        frame_counts = []
        label_list = []

        for idx in range(self.batch_size):
            video_file = dataset_index[self.cursor]
            video_path = os.path.join(dir_path, video_file)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            each_frames = []
            if not cap.isOpened():
                print(f"Error: Failed to open video file {video_path}")
                self.cursor += 1
                continue
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_frame, (self.crop_size, self.crop_size)) # 128x128 -> 64x64
                tensor_frame = to_tensor(resized) * 2 - 1
                each_frames.append(tensor_frame)
            # save each frames of one video
            # image save directory
            #save_dir = f"./videogan/saved_frames/{os.path.splitext(video_file)[0]}"
            #os.makedirs(save_dir, exist_ok=True)
            # save each frames
            #for frame_idx, frame_tensor in enumerate(each_frames):
            #    frame_tensor = (frame_tensor + 1) / 2.0
            #    pil_img = F.to_pil_image(frame_tensor)
            #    pil_img.save(os.path.join(save_dir, f"frame_{frame_idx:04d}.png")) 
            total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.cursor += 1

            if len(each_frames) == 0:
                print(f"Warning: No frames found in video {video_path}")
                continue

            video_tensor = torch.stack(each_frames, dim=0)  # shape: (T, 3, H, W) = [Frame, channel, height, width]
            batch_video_tensor_list.append(video_tensor)
            frame_counts.append(video_tensor.shape[0])
            frame_len = video_tensor.shape[0]

            if frame_len < self.frame_size:
                pad_len = self.frame_size - frame_len
                pad_tensor = torch.zeros((pad_len, 3, self.crop_size, self.crop_size))
                video_tensor = torch.cat([video_tensor, pad_tensor], dim=0)
            else:
                video_tensor = video_tensor[:self.frame_size]
            t_out[idx] = video_tensor  # (128, 3, 64, 64)
        
        #print("t_out : ", t_out.shape)
        # [Batch, Frame, channel, height, width]
        # (5, 128, 3, 64, 64)
        return t_out 
        
d = DataLoader().get_batch('train')