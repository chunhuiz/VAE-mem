import argparse

from model import resnet_i3d
from model.I3D import I3D_Base
import torch
import os 
import numpy as np
import h5py
import cv2 
from opencv_videovision import transforms
from tqdm import tqdm 
import train_utils

def load_model(model, pretrained_model_path):
    model_dict = torch.load(pretrained_model_path)
    state_dict = model.state_dict()
    new_dict = {k:v for k,v in model_dict.items() if k in state_dict.keys()}
    state_dict.update(new_dict)
    model.load_state_dict(state_dict)

def frame_processing(frames, tencrop=False):
    new_frames = []
    for frame in frames:
        img_decode=cv2.cvtColor(cv2.imdecode(np.frombuffer(frame,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        new_frames.append(img_decode)
    
    # mean = [114.75, 114.75, 114.75]
    # std = [57.375, 57.375, 57.375]
    mean = [128, 128, 128]
    std = [128, 128, 128]
    
    if tencrop:
        frame_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                    transforms.ClipToTensor(div_255=False),
                                    transforms.Normalize(mean=mean, std=std),
                                    transforms.TenCropTensor(224)
                                    ])
    else:
        frame_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                        transforms.ClipToTensor(div_255=False),
                                        transforms.Normalize(mean=mean, std=std)])
    new_frames = frame_transforms(new_frames)
    return new_frames

def flow_processing(flows, tencrop=False):
    flows = np.asarray(flows).astype(float)
    mean=[128,128,128]
    std=[128,128,128]
    
    if tencrop:
        flow_transforms = transforms.Compose([transforms.Resize((256,256)), 
                                            transforms.ClipToTensor(channel_nb=2, div_255=False),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.TenCropTensor(224),
                                            ])
    else:
        flow_transforms = transforms.Compose([transforms.Resize((224, 224)), 
                                            transforms.ClipToTensor(channel_nb=2, div_255=False),
                                            transforms.Normalize(mean=mean, std=std),
                                            ])
    return flow_transforms(flows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mode', default='rgb')
    parser.add_argument('--data_path', default='../VAD_datasets/SHT_Frames.h5')
    parser.add_argument('--save_path', default='../datasets/SHT_fea/')
    parser.add_argument('--h5_path', default= "./data/SHT_fea.h5")
    parser.add_argument('--tencrop', default=False)
    args = parser.parse_args()

    train_utils.set_gpu(args.gpu)

    # model = I3D(modality=args.mode)
    # if args.mode == 'rgb':
    #     load_model(model, "pretrained/model_rgb.pth")
    # elif args.mode == 'flow':
    #     load_model(model, "pretrained/model_flow.pth")
    # else:
    #     print(args.mode + " failed")
    
    # # I3D model
    # model = resnet_i3d.i3_res50(num_classes=400)
    # I3D nonlocal model
    # model = resnet_i3d.i3_res50_nl(num_classes=400)
    
    model = I3D_Base(0.8, 8, pretrained_backbone=True, rgb_model_path="pretrained/model-epoch-780-AUC-0.9703214147293845.pth",
                     flow_model_path="pretrained/model_flow.pth").cuda()

    model.cuda()
    model.eval()

    if os.path.exists(args.h5_path):
        os.remove(args.h5_path)
    
    
    dataset = h5py.File(args.data_path,'r')
    fea_dict_h5 = h5py.File(args.h5_path,"a")

    fea_dict = {}
    for key in tqdm(dataset.keys()):
        frames = [frame for frame in dataset[key]]
        video_name = key.split("-")[0]
        if video_name not in fea_dict.keys():
            fea_dict[video_name] = []

        if args.mode == 'rgb':
            frames = frame_processing(frames, args.tencrop)
        elif args.mode == 'flow':
            frames = flow_processing(frames, args.tencrop)
        else:
            print(args.mode + " failed")
            exit(1)

        if args.tencrop:
            frames = torch.stack(frames)
        frames = frames.view([-1, frames.shape[-4], frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda().float()
        fea = model.extract_features(frames,None)
        # pool =  torch.nn.AdaptiveAvgPool3d(1)
        # fea = pool(fea).squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu().numpy()
        fea = fea.squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu().numpy()
        fea_dict[video_name].append(fea)        
        # print(fea.shape)
        # break
        fea_dict_h5.create_dataset(key,data=fea,chunks=True)
        
    ### save as RTFM mode ###
    if not os.path.exists(args.save_path): 
        os.mkdir(args.save_path)
    for key, value in fea_dict.items():
        save_file = os.path.join(args.save_path, key+"_i3d.npy")
        value = np.stack(value)
        np.save(save_file, value, allow_pickle=True)
        print(key, value.shape)