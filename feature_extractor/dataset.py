import torch
from torch.utils.data import Dataset
import torchvision
from feature_extractor.opencv_videovision import transforms
import numpy as np
import h5py
import cv2
import os
import time
class SHT_C3D_feat_dataset(Dataset):
    def __init__(self,rgb_h5_file, flow_h5_file,segment_len=16,ten_crop=False,mode='rgb',height=128,width=171,crop_size=112):
        self.rgb_h5_file = rgb_h5_file
        self.flow_h5_file = flow_h5_file
        self.segment_len = segment_len
        self.ten_crop = ten_crop
        self.mode = mode
        self.crop_size = crop_size

        self.mean=[90.25,97.66,101.41]
        self.std=[1,1,1]
        self.height=height
        self.width=width

        self.keys=sorted(list(h5py.File(self.rgb_h5_file, 'r').keys()))

        if mode == 'rgb':
            self.h5_file = rgb_h5_file
            self.transforms = transforms.Compose([transforms.Resize([self.height, self.width]),
                                                  transforms.CenterCrop(self.crop_size),
                                                  transforms.ClipToTensor(div_255=False),
                                                  transforms.Normalize(mean=self.mean, std=self.std)])

        else:
            self.h5_file = flow_h5_file
            self.transforms = transforms.Compose([transforms.Resize([self.height, self.width]),
                                                  transforms.CenterCrop(self.crop_size),
                                                  transforms.ClipToTensor(channel_nb=2, div_255=False),
                                                  transforms.Normalize(mean=self.mean, std=self.std)])



    def __len__(self):
        return len(self.keys)

    def frame_processing(self, frames):
        if self.mode == 'rgb':
            new_frames = []
            for frame in frames:
                new_frames.append(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR))

            new_frames = self.transforms(new_frames)
        else:
            frames = np.asarray(frames).astype(float)

            new_frames = self.transforms(frames)
        return new_frames

    def decode_imgs(self, frames):
        new_frames = []
        for i, frame in enumerate(frames):
            new_frames.append(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR))

        # new_frames=torch.from_numpy(new_frames).float().permute([3,0,1,2])
        new_frames = self.transforms(new_frames)
        return new_frames

    def __getitem__(self, i):
        key = self.keys[i]
        # frames=h5py.File(self.h5_path,'r')[key][:]
        with h5py.File(self.h5_file, 'r') as frame_h5:
            frames = frame_h5[key][:]

        # begin=time.time()
        frames = self.frame_processing(frames)
        # end=time.time()
        # print('take time {}'.format(end-begin))
        return key, frames

class SHT_I3D_feat_dataset(Dataset):
    def __init__(self,rgb_h5_file, flow_h5_file, segment_len=16, ten_crop=False, mode='rgb', height=256, width=340, crop_size=224):
        self.rgb_h5_file = rgb_h5_file
        self.flow_h5_file = flow_h5_file

        # self.keys = sorted(list(h5py.File(self.h5_path, 'r').keys()))
        self.segment_len = segment_len
        self.ten_crop = ten_crop
        self.crop_size = crop_size
        self.mode = mode
        self.mean=[128,128,128]
        self.std=[128,128,128]
        self.height=height
        self.width=width

        self.keys=sorted(list(h5py.File(self.rgb_h5_file, 'r').keys()))


        # if ten_crop:
        #     self.transforms = transforms.Compose([transforms.Resize([240, 320]),
        #                                             transforms.ClipToTensor(div_255=False),
        #                                             transforms.Normalize(mean=self.mean,std=self.std),
        #                                             transforms.TenCropTensor(224)])

        if mode == 'rgb':
            self.h5_file = rgb_h5_file
            self.transforms = transforms.Compose([ transforms.Resize([256, 256]),
                                                transforms.CenterCrop(self.crop_size),
                                                transforms.ClipToTensor(div_255=False),
                                                transforms.Normalize(mean=self.mean, std=self.std)])

        else:
            self.h5_file = flow_h5_file
            self.transforms = transforms.Compose([transforms.Resize([256, 256]),
                                                  transforms.CenterCrop(self.crop_size),
                                                  transforms.ClipToTensor(channel_nb=2, div_255=False),
                                                  transforms.Normalize(mean=self.mean, std=self.std)])


    def __len__(self):
        return len(self.keys)

    def frame_processing(self, frames):
        if self.mode == 'rgb':
            new_frames = []
            for frame in frames:
                img_decode = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                new_frames.append(img_decode)
            new_frames = self.transforms(new_frames)
        else:
            frames = np.asarray(frames).astype(float)

            new_frames = self.transforms(frames)
        return new_frames

    def __getitem__(self, i):
        key = self.keys[i]
        with h5py.File(self.h5_file, 'r') as frame_h5:
            frames = frame_h5[key][:]

        # begin=time.time()
        frames = self.frame_processing(frames)

        if self.ten_crop:
            frames = torch.stack(frames)
        # end=time.time()
        # print('take time {}'.format(end-begin))
        return key, frames




