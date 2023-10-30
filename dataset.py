import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import h5py
import cv2
import os
from train_utils import random_perturb
import time

Abnormal_type=['Abuse','Arrest','Arson','Assault','Burglary',
               'Explosion','Fighting','RoadAccidents','Robbery',
               'Shooting','Shoplifting','Stealing','Vandalism','Normal']


class SHT_LPN_Test_dataset(Dataset):
    def __init__(self,rgb_h5_file, test_txt, test_mask_dir,
                 segment_len=16, ten_crop=False):
        self.rgb_h5_file = list(open(rgb_h5_file))[0]
        # self.keys = sorted(list(h5py.File(self.h5_path, 'r').keys()))
        self.test_txt=test_txt
        self.test_mask_dir=test_mask_dir
        self.segment_len = segment_len
        self.ten_crop = ten_crop

        self.test_dict_annotation()

    def __len__(self):
        return len(self.video_keys)

    def test_dict_annotation(self):
        self.annotation_dict = {}
        self.keys=[]
        self.video_keys=[]

        keys=sorted(list(h5py.File(self.rgb_h5_file, 'r').keys()))
        for line in open(self.test_txt,'r').readlines():
            key,anno_type,frames_num = line.strip().split(',')
            self.video_keys.append(key)
            # frames_num=int(frames_num)
            frames_num = 0
            for h5_key in keys:
                if h5_key.split('-')[0] == key:
                    frames_num = frames_num + 1
            if anno_type=='1':
                label='Abnormal'
                anno = np.load(os.path.join(self.test_mask_dir, key + '.npy'))#[
                       #:frames_num - frames_num % self.segment_len]
            else:
                label='Normal'
                anno=np.zeros(frames_num * self.segment_len,dtype=np.uint8)
            self.annotation_dict[key]=[anno,label]
        print(self.video_keys)
        print(len(self.video_keys))
        # key_dict={}
        # for key in keys:
        #     if key.split('-')[0] in self.annotation_dict.keys():
        #         self.keys.append(key)
        # print(len(self.keys))

        # np.save('SHT_Test_keys.npy', self.keys)

    def __getitem__(self, i):
        video_key = self.video_keys[i]
        anno = self.annotation_dict[video_key][0].astype(np.uint8)
        video_len = len(anno) // 16
        frames = []
        with h5py.File(self.rgb_h5_file, 'r') as rgb_h5:
            for i in range(video_len):
                frames.append(torch.from_numpy(rgb_h5[video_key + '-{0:06d}'.format(i)][:]))

        frames = torch.stack(frames)
        anno = anno[:video_len*16]
        return frames

        # return frames, anno


class SHT_LPN_Train_dataset(Dataset):
    def __init__(self, rgb_h5_file, train_txt, clip_num=32, segment_len=16,
                 type='Normal', ten_crop=False,continuous_sampling=False, real_label=False):
        self.rgb_h5_file = list(open(rgb_h5_file))[0]
        self.train_txt=train_txt
        self.clip_num=clip_num
        self.segment_len = segment_len
        self.type = type
        self.ten_crop = ten_crop
        self.continuous_sampling=continuous_sampling
        self.real_label = real_label
        self.keys = sorted(list(h5py.File(self.rgb_h5_file, 'r').keys()))

        self.get_vid_names_dict()
        self.test_mask_dir = 'data/SHT/test_frame_mask'
        self.test_dict_annotation()

        if self.type == 'Normal':
            self.selected_keys = list(self.norm_vid_names_dict.keys())
            self.selected_dict=self.norm_vid_names_dict
        else:
            self.selected_keys = list(self.abnorm_vid_names_dict.keys())
            self.selected_dict=self.abnorm_vid_names_dict


    def __len__(self):
        return len(self.selected_keys)

    def test_dict_annotation(self):
        self.annotation_dict = {}
        keys=sorted(list(h5py.File(self.rgb_h5_file, 'r').keys()))

        for line in open(self.train_txt,'r').readlines():
            key,anno_type = line.strip().split(',')
            frames_num = 0
            for h5_key in keys:
                if h5_key.split('-')[0] == key:
                    frames_num = frames_num + 1

            frames_num = frames_num * 16
            if anno_type=='1':
                label='Abnormal'
                anno = np.load(os.path.join(self.test_mask_dir, key + '.npy'))#[
                       #:frames_num - frames_num % self.segment_len]
            else:
                label='Normal'
                anno=np.zeros(frames_num,dtype=np.uint8)

            self.annotation_dict[key]=[anno,label]


    def get_vid_names_dict(self):
        self.norm_vid_names_dict = {}
        self.abnorm_vid_names_dict = {}

        for line in open(self.train_txt,'r').readlines():
            key,label=line.strip().split(',')
            if label=='1':
                for k in self.keys:
                    if key == k.split('-')[0]:
                        if key in self.abnorm_vid_names_dict.keys():
                            self.abnorm_vid_names_dict[key]+=1
                        else:
                            self.abnorm_vid_names_dict[key]=1
            else:
                for k in self.keys:
                    if key == k.split('-')[0]:
                        if key in self.norm_vid_names_dict.keys():
                            self.norm_vid_names_dict[key]+=1
                        else:
                            self.norm_vid_names_dict[key]=1

    def __getitem__(self, i):

        key = self.selected_keys[i]
        scores = self.annotation_dict[key][0]
        video_len = len(scores)
        if self.real_label:
            # 载入真实标签
            scores = scores[ : len(scores) - len(scores) % 16]
            scores = scores.reshape((-1, 16))
            # scores = np.mean(scores, 1)
            scores = np.max(scores, 1)
        else:
            if self.type == 'Normal':
                scores = np.zeros(video_len, dtype=np.uint8)
            else:
                scores = np.ones(video_len, dtype=np.uint8)

        vid_len=self.selected_dict[key]

        if not self.continuous_sampling:
            chosens = random_perturb(vid_len-1, self.clip_num)
        else:
            chosens= np.random.randint(0,vid_len-1-self.clip_num)+np.arange(0, self.clip_num)
        labels=[]
        rgb_clips = []

        with h5py.File(self.rgb_h5_file, 'r') as rgb_h5:
            for chosen in chosens:
                labels.append(scores[chosen])
                rgb_clips.append(torch.from_numpy(rgb_h5[key + '-{0:06d}'.format(chosen)][:]))

        rgb_clips=torch.stack(rgb_clips)

        return rgb_clips, np.array(labels)


class SHT_Dataset(Dataset):
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
                self.list = self.list[63:]
                print('normal list')
                # print(self.list)
            else:
                self.list = self.list[:63]

                print('abnormal list')
                # print(self.list)

    def __getitem__(self, index):

        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)  # [T, 10, F]

        crop_idx = np.random.randint(0, 10, [1])
        # features = features[:, crop_idx].squeeze(1)  # [T,F]

        features = features.transpose(1, 0, 2)  # [10, T, F]
        if self.tranform is not None:
            features = self.tranform(features)

        if self.test_mode:
            return torch.from_numpy(features)
        else:
            vid_len = features.shape[1]
            chosens = random_perturb(vid_len - 1, self.clip_num)
            if self.is_normal == 'Normal':
                labels = np.zeros(len(chosens), dtype=np.uint8)
            else:
                labels = np.ones(len(chosens), dtype=np.uint8)
            rgb_clips = []
            for chosen in chosens:
                rgb_clips.append(torch.from_numpy(features[:, chosen]))
            rgb_clips = torch.stack(rgb_clips).permute(1, 0, 2)
            # print(rgb_clips.shape)
            return rgb_clips, torch.from_numpy(np.array(labels))

            # divided_features = []
            # for feature in features:
            #     feature = process_feat(feature, 32)  # divide a video into 32 segments
            #     divided_features.append(feature)
            # divided_features = np.array(divided_features, dtype=np.float32)
            #
            # return torch.from_numpy(divided_features), torch.from_numpy(np.array(labels))

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame


class UCF_Dataset(Dataset):
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
        print(self.list)
    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[810:]
                print('normal list')
                # print(self.list)
            else:
                self.list = self.list[:810]

                print('abnormal list')
                # print(self.list)

    def __getitem__(self, index):

        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)  # [T, 10, F]

        crop_idx = np.random.randint(0, 10, [1])
        # features = features[:, crop_idx].squeeze(1)  # [T,F]

        features = features.transpose(1, 0, 2)  # [10, T, F]
        if self.tranform is not None:
            features = self.tranform(features)

        if self.test_mode:
            return torch.from_numpy(features)
        else:
            vid_len = features.shape[1]
            chosens = random_perturb(vid_len - 1, self.clip_num)
            if self.is_normal:
                labels = np.zeros(len(chosens), dtype=np.uint8)
            else:
                labels = np.ones(len(chosens), dtype=np.uint8)
            rgb_clips = []
            for chosen in chosens:
                rgb_clips.append(torch.from_numpy(features[:, chosen]))
            rgb_clips = torch.stack(rgb_clips).permute(1, 0, 2)
            # print(rgb_clips.shape)
            return rgb_clips, torch.from_numpy(np.array(labels))

            # divided_features = []
            # for feature in features:
            #     feature = process_feat(feature, 32)  # divide a video into 32 segments
            #     divided_features.append(feature)
            # divided_features = np.array(divided_features, dtype=np.float32)
            #
            # return torch.from_numpy(divided_features), torch.from_numpy(np.array(labels))

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame

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
            vid_len = features.shape[0]

            chosens = random_perturb(vid_len - 1, self.clip_num)

            if self.is_normal:
                labels = np.zeros(len(chosens), dtype=np.uint8)
            else:
                labels = np.ones(len(chosens), dtype=np.uint8)

            rgb_clips = []
            for chosen in chosens:

                rgb_clips.append(torch.from_numpy(features[chosen]))

            rgb_clips = torch.stack(rgb_clips)

            return rgb_clips, torch.from_numpy(np.array(labels))

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
