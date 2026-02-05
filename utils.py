import imageio
import numpy as np
from torchvision.transforms import ToPILImage
import logging
import sys
from torchvision.transforms import ToPILImage
from PIL import Image
import logging
import sys
import imageio
import PIL
import PIL.Image
import skvideo.io
import skimage.transform
import numpy as np

def process_and_write_image(images,name):
    images = np.array(images).transpose((0,2,3,4,1))
    images = (images + 1)*127.5
    for i in range(images.shape[0]):
        PIL.Image.fromarray(np.around(images[i,0,:,:,:]).astype(np.uint8)).save("./genvideos/" + name + ".jpg")

def read_and_process_video(files,size,nof):
    videos = np.zeros((size,nof,64,64,3))
    counter = 0
    for file in files:
        vid = skvideo.io.vreader(file)
        curr_frames = []
        i = 0
        
        nr = np.random.randint(20)
        for frame in vid:
            i = i + 1
            if i <= nr:
                continue

            frame = skimage.transform.resize(frame,[64,64])
            curr_frames.append(frame)

            if i >= nr+nof:
                break

        curr_frames = np.array(curr_frames)
        curr_frames = curr_frames*255.0
        curr_frames = curr_frames/127.5 - 1
        videos[counter,:,:,:,:] = curr_frames
        counter = counter + 1

    return videos.transpose((0,4,1,2,3)).astype(np.float32)

def process_and_write_video(videos,name):
    videos = np.array(videos)
    videos = np.reshape(videos,[-1,3,32,64,64]).transpose((0,2,3,4,1))
    vidwrite = np.zeros((32,64,64,3))
    for i in range(videos.shape[0]):
        vid = videos[i,:,:,:,:]
        vid = (vid + 1)*127.5
        for j in range(vid.shape[0]):
            frame = vid[j,:,:,:]
            vidwrite[j,:,:,:] = frame
        skvideo.io.vwrite("./genvideos/" +name + ".mp4",vidwrite)


# Receives [3,32,64,64] tensor, and creates a gif
def make_gif(images, filename):
    # images shape: [3, 32, 64, 64]  (channels, frames, H, W)
    # permute to (frames, H, W, channels)
    x = images.permute(1, 2, 3, 0).cpu().numpy()  # shape: (32,64,64,3)
    
    # 이미지 값이 float일 경우 [0,1] 범위로 가정하고 uint8 변환
    x = np.clip(x, 0, 1)  # 혹시 범위 벗어나면 클리핑
    x = (x * 255).astype(np.uint8)

    frames = []
    for i in range(x.shape[0]):
        frames.append(x[i])
    imageio.mimsave(filename, frames)

def denorm(x):
    return (x + 1.0) / 2.0

def save_img(x, filename): 
    x = denorm(x)
    x = x.squeeze()
    to_pil = ToPILImage()
    img = to_pil(x)
    img.save(filename)
    
'''
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
'''
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
    
'''
def make_gif(images, filename):
    x = images.permute(1,2,3,0)
    x = x.numpy()
    frames = []
    for i in range(32):
        frames += [x[i]]
    imageio.mimsave(filename, frames)
'''
#x = torch.rand((3,32,64,64))
#make_gif(x, 'movie.gif')
