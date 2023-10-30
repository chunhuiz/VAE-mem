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
from model.memory import Memory
from losses import Mem_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embedding(data, label, title, save_path):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 5.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xlim(-0.4, 1.4)
    plt.ylim(-0.4, 1.4)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(save_path)

def eval_epoch(config, model, test_dataloader, nor_keys, abn_keys, epoch):
    model = model.eval()
    total_labels, total_scores = [], []
    i = 0
    frame_num = 0
    gt = list(np.load(config['gt']))

    for frames in tqdm(test_dataloader):
        i += 1
        frames = frames.view(-1, frames.shape[-2], frames.shape[-1])
        # frames=frames.float().contiguous().view([-1, 1, frames.shape[-1]]).cuda()
        # flows=flows.float().contiguous().view([-1, 1, flows.shape[-1]]).cuda()
        with torch.no_grad():
            score, updated_score, nor_compactness_loss, abn_compactness_loss, feat = model(frames, frames, nor_keys, abn_keys, epoch, isTrain=False, gl=config['test_gl'])
            # print(score.shape)
            if config['dataset_name'] == 'UCF':
                score = torch.mean(score, dim=0)
                nor_compactness_loss = nor_compactness_loss.view(10, -1)
                nor_compactness_loss = torch.mean(nor_compactness_loss, dim=0)
                abn_compactness_loss = abn_compactness_loss.view(10, -1)
                abn_compactness_loss = torch.mean(abn_compactness_loss, dim=0)
            tmp = nor_compactness_loss / abn_compactness_loss
            extra_score = (2 * tmp - 1) / (2 - 1*tmp)
            # print(score)
            # print(nor_compactness_loss)
            # print(abn_compactness_loss)
            # print(extra_score)
            f_score = (score * extra_score).clamp(0., 1.)
            f_score = list(f_score.cpu().detach().numpy())
            f_score = np.repeat(np.array(f_score), 16)
            frame_num_before = frame_num
            frame_num += len(f_score)
            total_scores.extend(f_score)

            if epoch > 20000:
                path = os.path.join(config['save_path'], f'{epoch}')
                if not os.path.exists(path):
                    os.makedirs(path)
                print(feat.shape)
                feat = feat.reshape(-1, feat.shape[-1])
                keys = torch.cat([nor_keys, abn_keys, feat], 0)
                label = [0] * len(nor_keys) + [1] * len(abn_keys) + [2] * len(feat)
                tsne = TSNE(n_components=2, init='pca', random_state=0)
                result = tsne.fit_transform(keys.cpu().numpy())
                save_path = os.path.join(path, f'{i}.jpg')
                plot_embedding(result, label,
                               't-SNE embedding of memory keys', save_path)

            if True and epoch >= 5 and epoch % 5 == 0:
                gt_now = gt[frame_num_before:frame_num]
                if max(gt_now) >= 1:

                    save_path_score = os.path.join(config['save_path'], 'score')
                    if not os.path.exists(save_path_score):
                        os.makedirs(save_path_score)
                    # 绘制psnr
                    plt.figure(figsize=(10,4))
                    plt.xlabel('Frame Index')
                    plt.ylabel('Anomaly Score')
                    min_ = 0
                    max_ = 1.1
                    plt.ylim(min_, max_)
                    plt.plot(f_score)


                    # 绘制真值
                    # plt.plot(gt_now, color='r')
                    plt.fill_between(np.linspace(0, len(gt_now), len(gt_now)),
                                     0, gt_now, facecolor='r', alpha=0.3)
                    plt.savefig(os.path.join(save_path_score, f'{epoch}_{i}.jpg'))



    if config['dataset_name'] == 'XDViolence':
        total_scores = np.array(total_scores).reshape([-1, 5])
        total_scores = np.mean(total_scores, axis=1)




    if True:
        save_path = os.path.join(config['save_path'], f'{epoch}.jpg')
        keys = torch.cat([nor_keys, abn_keys], 0)
        label = [0] * len(nor_keys) + [1] * len(abn_keys)
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(keys.cpu().numpy())
        plot_embedding(result, label,
                       't-SNE embedding of memory keys', save_path)


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
    model = Memory(config['feature_dim'], config['out_feature_dim'], config['layer_num'], config['dropout_rate']).cuda()

    # optimizer setting
    params = list(model.parameters())
    optimizer, lr_scheduler = train_utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])

    model = torch.nn.parallel.DataParallel(model, dim=0, device_ids=config['gpu'])

    model = model.train()
    criterion = Mem_loss(config).cuda()

    train_utils.log('Start train')
    iterator = 0
    test_epoch = 10 if config['eval_epoch'] is None else config['eval_epoch']
    AUCs,tious,best_epoch,best_tiou_epoch,best_tiou,best_AUC=[],[],0,0,0,0

    ref_iter_abn = iter(ref_dataloader_abn)
    nor_keys = F.normalize(torch.rand((10, config['out_feature_dim']), dtype=torch.float),
                       dim=1).cuda()
    abn_keys = F.normalize(torch.rand((10, config['out_feature_dim']), dtype=torch.float),
                           dim=1).cuda()
    auc = eval_epoch(config, model, test_dataloader, nor_keys, abn_keys, 0)
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

            p_score, updated_p_score, label, nor_keys, abn_keys, separateness_loss, compactness_loss, inter_loss = model(ref_rgb_nor, ref_rgb_abn, nor_keys, abn_keys, epoch)
            cost = criterion(p_score, updated_p_score, label)
            # cost = cost + 0.1 * separateness_loss + 0.1 * compactness_loss
            # print('-------')
            # print(cost)
            # print(separateness_loss)
            # print(compactness_loss)
            # print(inter_loss)

            cost = cost + 0.1 * separateness_loss + 0.1 * compactness_loss + 0.1 * inter_loss
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

        if epoch % test_epoch == 0 and epoch > 5:

            auc = eval_epoch(config, model, test_dataloader, nor_keys, abn_keys, epoch)
            AUCs.append(auc)
            if len(AUCs) >= 5:
                mean_auc = sum(AUCs[-5:]) / 5.
                if mean_auc > best_AUC:
                    best_epoch,best_AUC =epoch,mean_auc
                train_utils.log('best_AUC {} at epoch {}, now {}'.format(best_AUC, best_epoch, mean_auc))
            train_utils.log(nor_keys.shape)
            train_utils.log(abn_keys.shape)
            train_utils.log('===================')
            if auc > 0.971:
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
