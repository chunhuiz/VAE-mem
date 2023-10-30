import os
import sys
sys.path.append('..')
import h5py
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import time

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from feature_extractor.dataset import SHT_I3D_feat_dataset, SHT_C3D_feat_dataset
from feature_extractor.I3D import I3D
from feature_extractor.C3D import C3D

def worker_init(worked_id):
    np.random.seed(worked_id)
    random.seed(worked_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', default='SHT')

    parser.add_argument('--mode', default='rgb')
    parser.add_argument('--structure', default='C3D')


    parser.add_argument('--train_split', default='../data/SH_Train_new.txt')
    parser.add_argument('--test_split', default='../data/SH_Test_NEW.txt')
    parser.add_argument('--segment_len', default=16)
    parser.add_argument('--ten_crop', action='store_true')
    # parser.add_argument('--rgb_model_path', default='/data0/JY/lyx/WVAD/pretrained/model_rgb.pth')
    # parser.add_argument('--rgb_model_path', default='../pretrained/model-AUC-0.97.pth')
    # parser.add_argument('--rgb_model_path', default='../pretrained/UCF-I3D-model-AUC-0.805.pth')

    # parser.add_argument('--flow_model_path', default='/data0/JY/lyx/WVAD/model_flow.pth')
    # parser.add_argument('--h5_path', default= "/data2/Datasets/UCF_fea_refine.h5")

    args = parser.parse_args()

    if args.dataset == 'SHT':
        rgb_dataset_path = '/data0/JY/zwh/Datasets/SHT_Frames.h5'
        flow_dataset_path = '/data0/JY/zwh/Datasets/SHT_Flows.h5'
        # rgb_model_path = '../pretrained/model-AUC-0.97.pth'
        rgb_model_path = '/data0/JY/zwh/WVAD-hard-relation/feature_extractor/C3D-model-AUC-0.94.pth'
        # rgb_model_path = '/data0/JY/lyx/WVAD/pretrained/model_rgb.pth'

        # flow_model_path = '/data0/JY/lyx/WVAD/pretrained/model_flow.pth'
        h5_path = '/data2/Datasets/SHT_fea_'+ args.mode+'_'+args.structure+'_refine.h5'
    else:
        rgb_dataset_path = '/data0/JY/lyx/datasets/UCF-Frames.h5'
        flow_dataset_path = ''
        # rgb_model_path = '../pretrained/UCF-I3D-model-AUC-0.805.pth'
        rgb_model_path = '/data0/JY/lyx/WVAD/pretrained/model_rgb.pth'
        flow_model_path = '/data0/JY/lyx/WVAD/pretrained/model_flow.pth'
        h5_path = '/data2/Datasets/UCF_fea_ori_'+ args.mode+'_refine.h5'



    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus =[i for i in range(len(args.gpu.split(',')))]

    if args.structure == 'I3D':

        model = I3D(modality=args.mode).cuda()
    else:
        model = C3D().cuda()

    if args.mode == 'rgb':
        pretrained_dict = torch.load(rgb_model_path)['model']
    else:
        pretrained_dict = torch.load(flow_model_path)

    state_dict = model.state_dict()
    print(state_dict.keys())
    print(pretrained_dict.keys())
    # print('--------')

    new_dict = {k[len('module.backbone.'):]: v for k, v in pretrained_dict.items() if k[len('module.backbone.'):] in state_dict.keys()}
    # new_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict.keys()}
    # new_dict = {k[len('module.rgb_backbone.'):]: v for k, v in pretrained_dict.items() if k[len('module.rgb_backbone.'):] in state_dict.keys()}
    state_dict.update(new_dict)
    model.load_state_dict(new_dict)

    model = torch.nn.parallel.DataParallel(model, dim=0, device_ids=gpus).eval()

    if args.structure == 'I3D':
        i3d_dataset_rgb = SHT_I3D_feat_dataset(rgb_dataset_path, flow_dataset_path,
                                               segment_len=args.segment_len, ten_crop=args.ten_crop, mode=args.mode)
        dataloader = DataLoader(i3d_dataset_rgb, batch_size=1, shuffle=False,
                                num_workers=10, worker_init_fn=worker_init, drop_last=False, pin_memory=True)
    else:
        c3d_dataset_rgb = SHT_C3D_feat_dataset(rgb_dataset_path, flow_dataset_path,
                                               segment_len=args.segment_len, ten_crop=args.ten_crop, mode=args.mode)
        dataloader = DataLoader(c3d_dataset_rgb, batch_size=1, shuffle=False,
                                        num_workers=10, worker_init_fn=worker_init, drop_last=False, pin_memory=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)
    fea_dict_h5 = h5py.File(h5_path, "a")

    fea_dict = {}
    print('start!')
    for key, frames in tqdm(dataloader):
        # [1, 3, 16, 256, 256]
        key = key[0]
        if args.mode == 'rgb':
            frames = frames.float().contiguous().view(
                [-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()
        else:
            frames = frames.float().contiguous().view(
                [-1, 2, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()
        # video_name = key.split("-")[0]
        # if video_name not in fea_dict.keys():
        #     fea_dict[video_name] = []

        with torch.no_grad():
            feat = model(frames)[0].squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu().numpy()

        fea_dict_h5.create_dataset(key,data=feat,chunks=True)




