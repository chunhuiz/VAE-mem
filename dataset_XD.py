import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import h5py
import cv2
import os
from train_utils import random_perturb
import time


def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]

def uniform_extract(feat, t_max):
   r = np.linspace(0, len(feat)-1, t_max, dtype=np.uint16)
   return feat[r, :]

def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=True):
    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)

class XD_Dataset(Dataset):
    def __init__(self, rgb_list, test_rgb_list, clip_num, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = test_rgb_list
        else:
            self.rgb_list_file = rgb_list
        self.clip_num = clip_num
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[9525:]
                print('normal list')
                # print(self.list)
            else:
                self.list = self.list[:9525]

                print('abnormal list')
                # print(self.list)

    def __getitem__(self, index):

        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)

        if self.test_mode:
            return torch.from_numpy(features)
        else:

            # features = process_feat(features, self.clip_num, is_random=False)

            vid_len = features.shape[0]
            chosens = random_perturb(vid_len - 1, self.clip_num)
            rgb_clips = []
            for chosen in chosens:
                rgb_clips.append(torch.from_numpy(features[chosen]))
            rgb_clips = torch.stack(rgb_clips)

            if self.is_normal:
                labels = np.zeros(1, dtype=np.uint8)
            else:
                labels = np.ones(1, dtype=np.uint8)

            # if '_label_A' in self.list[index]:
            #     labels = np.zeros(1, dtype=np.uint8)
            # else:
            #     labels = np.ones(1, dtype=np.uint8)

            return rgb_clips, torch.from_numpy(np.array(labels))

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
