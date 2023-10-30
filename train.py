import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import time

import torch
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter

import train_utils
from train_utils import AverageMeter
from eval_utils import eval, cal_rmse

from dataset import SHT_LPN_Train_dataset, SHT_LPN_Test_dataset, UCF_Dataset, XD_Dataset
from model.RAPN import RAPN
from losses import Total_loss


def eval_epoch(config, model, test_dataloader):
    model = model.eval()
    total_labels, total_scores = [], []

    for frames in tqdm(test_dataloader):

        frames = frames.view(-1, frames.shape[-2], frames.shape[-1])
        # frames=frames.float().contiguous().view([-1, 1, frames.shape[-1]]).cuda()
        # flows=flows.float().contiguous().view([-1, 1, flows.shape[-1]]).cuda()
        with torch.no_grad():
            score = model(frames, frames, isTrain=False, gl=config['test_gl'])
            if config['dataset_name'] == 'UCF':
                score = torch.mean(score, dim=0)
            score = list(score.cpu().detach().numpy())
            score = np.repeat(np.array(score), 16)
            total_scores.extend(score)

    if config['dataset_name'] == 'XDViolence':
        total_scores = np.array(total_scores).reshape([-1, 5])
        total_scores = np.mean(total_scores, axis=1)

    gt = list(np.load(config['gt']))
    return eval(total_scores, gt)


def train(config):
    #### set the save and log path ####
    save_path = config['save_path']
    train_utils.set_save_path(save_path)
    train_utils.set_log_path(save_path)
    #writer = SummaryWriter(os.path.join(config['save_path'], 'tensorboard'))
    yaml.dump(config, open(os.path.join(config['save_path'], 'classifier_config.yaml'), 'w'))

    #### make datasets ####
    def worker_init(worked_id):
        np.random.seed(worked_id)
        random.seed(worked_id)

    # train
    if config['dataset_name'] == 'SHT':
        ref_dataset_nor = SHT_LPN_Train_dataset(config['rgb_dataset_path'], config['train_split'],
                                                config['clips_num'], segment_len=config['segment_len'], type='Normal', ten_crop=config['ten_crop'], real_label=False)
        ref_dataset_abn = SHT_LPN_Train_dataset(config['rgb_dataset_path'],config['train_split'],
                                            config['clips_num'],segment_len=config['segment_len'], type='Abnormal', ten_crop=config['ten_crop'], real_label=False)

        test_dataset = SHT_LPN_Test_dataset(config['rgb_dataset_path'], config['test_split'],
                                        config['test_mask_dir'], segment_len=config['segment_len'],
                                        ten_crop=config['ten_crop'])
    elif config['dataset_name'] == 'UCF':
        ref_dataset_nor = UCF_Dataset(config['rgb_dataset_list'], config['test_rgb_dataset_list'],
                                  clip_num=config['clips_num'], is_normal=True)
        ref_dataset_abn = UCF_Dataset(config['rgb_dataset_list'], config['test_rgb_dataset_list'],
                                  clip_num=config['clips_num'], is_normal=False)

        test_dataset = UCF_Dataset(config['rgb_dataset_list'], config['test_rgb_dataset_list'],
                               clip_num=config['clips_num'], test_mode=True)
    elif config['dataset_name'] == 'XDViolence':
        ref_dataset_nor = XD_Dataset(config['rgb_dataset_list'], config['test_rgb_dataset_list'], clip_num=config['clips_num'], is_normal=True)
        ref_dataset_abn = XD_Dataset(config['rgb_dataset_list'], config['test_rgb_dataset_list'], clip_num=config['clips_num'], is_normal=False)

        test_dataset = XD_Dataset(config['rgb_dataset_list'], config['test_rgb_dataset_list'], clip_num=config['clips_num'], test_mode=True)


    ref_dataloader_nor = DataLoader(ref_dataset_nor, batch_size=config['batch_size'], shuffle=True,
                                 num_workers=16, worker_init_fn=worker_init, drop_last=True, pin_memory=True)
    ref_dataloader_abn = DataLoader(ref_dataset_abn, batch_size=config['batch_size'], shuffle=True,
                                    num_workers=16, worker_init_fn=worker_init, drop_last=True, pin_memory=True)

    # test
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False,
                                 num_workers=8, worker_init_fn=worker_init, drop_last=False, pin_memory=True)

    #### Model setting ####
    model = RAPN(config['feature_dim'], config['out_feature_dim'], config['layer_num'], config['dropout_rate']).cuda()

    # optimizer setting
    params = list(model.parameters())
    optimizer, lr_scheduler = train_utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])

    model = torch.nn.parallel.DataParallel(model, dim=0, device_ids=config['gpu'])

    model = model.train()
    criterion = Total_loss(config).cuda()

    train_utils.log('Start train')
    iterator = 0
    test_epoch = 10 if config['eval_epoch'] is None else config['eval_epoch']
    AUCs,tious,best_epoch,best_tiou_epoch,best_tiou,best_AUC=[],[],0,0,0,0

    ref_iter_abn = iter(ref_dataloader_abn)
    auc = eval_epoch(config, model, test_dataloader)
    print(auc)

    for epoch in range(config['epochs']):
        model = model.train()

        Errs, Atten_Errs, Rmses = AverageMeter(), AverageMeter(), AverageMeter()
        for step, (ref_rgb_nor, ref_labels_nor) in tqdm(enumerate(ref_dataloader_nor)):

            try:
                ref_rgb_abn, ref_labels_abn = next(ref_iter_abn)
            except:
                del ref_iter_abn
                ref_iter_abn = iter(ref_dataloader_abn)
                ref_rgb_abn, ref_labels_abn = next(ref_iter_abn)

            ref_rgb_nor, ref_labels_nor= ref_rgb_nor.cuda().float(), ref_labels_nor.cuda().float()
            ref_rgb_abn, ref_labels_abn = ref_rgb_abn.cuda().float(), ref_labels_abn.cuda().float()

            ref_rgb_nor = ref_rgb_nor.view(-1, ref_rgb_nor.shape[-2], ref_rgb_nor.shape[-1])
            ref_rgb_abn = ref_rgb_abn.view(-1, ref_rgb_abn.shape[-2], ref_rgb_abn.shape[-1])

            p_score, label, abn_score_topK_nor, abn_score_topK_abn, nor_feat_conf_nor, abn_feat_conf_nor, abn_feat_conf_abn \
                = model(ref_rgb_nor, ref_rgb_abn)
            p_label = torch.cat([ref_labels_nor, ref_labels_abn], dim=0)
            cost = criterion(p_score, p_label, label, abn_score_topK_nor, abn_score_topK_abn, nor_feat_conf_nor, abn_feat_conf_nor, abn_feat_conf_abn)

            cost.backward()

            optimizer.step()
            optimizer.zero_grad()
            Errs.update(cost)

            iterator += 1

            # if iterator % 10 == 0 and epoch > 14:
            #
            #     auc = eval_epoch(config, model, test_dataloader)
            #     AUCs.append(auc)
            #     if len(AUCs) >= 5:
            #         mean_auc = sum(AUCs[-5:]) / 5.
            #         if mean_auc > best_AUC:
            #             best_epoch, best_AUC = epoch, mean_auc
            #         train_utils.log('best_AUC {} at epoch {}, now {}'.format(best_AUC, best_epoch, mean_auc))
            #
            #     train_utils.log('===================')
            #     if auc > 0.84:
            #         checkpoint = {
            #             'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #         }
            #         torch.save(checkpoint,
            #                    os.path.join(save_path, 'models/model-epoch-{}-AUC-{}.pth'.format(epoch, auc)))

        train_utils.log('[{}]: err\t{:.4f}\tatten\t{:.4f}'.format(epoch, Errs, Atten_Errs))
        Errs.reset(), Atten_Errs.reset()

        train_utils.log("epoch {}, lr {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_utils.log('----------------------------------------')

        if epoch % test_epoch == 0 and epoch > 4:

            auc = eval_epoch(config, model, test_dataloader)
            AUCs.append(auc)
            if len(AUCs) >= 5:
                mean_auc = sum(AUCs[-5:]) / 5.
                if mean_auc > best_AUC:
                    best_epoch,best_AUC =epoch,mean_auc
                train_utils.log('best_AUC {} at epoch {}, now {}'.format(best_AUC, best_epoch, mean_auc))

            train_utils.log('===================')
            if auc > 0.8:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_path, 'models/model-epoch-{}-AUC-{}.pth'.format(epoch, auc)))

    train_utils.log('Training is finished')
    train_utils.log('max_frame_AUC: {}'.format(best_AUC))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if args.tag is not None:
        config['save_path'] += ('_' + args.tag)

    train_utils.set_gpu(args.gpu)
    config['gpu'] =[i for i in range(len(args.gpu.split(',')))]

    train(config)
