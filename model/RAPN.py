import math
import os
import torch
import torch.nn as nn
import train_utils
import numpy as np
from model.modules import Sample_generator, LPN


class Classifier(nn.Module):

    def __init__(self,feature_dim,dropout_rate=0.7):
        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(feature_dim,512), nn.ReLU(), nn.Dropout(dropout_rate),
        #                              nn.Linear(512,128), nn.ReLU(), nn.Dropout(dropout_rate),
        #                              nn.Linear(128,1), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(feature_dim, 1), nn.Sigmoid())

    def forward(self, x):
        # （bs, T, F)
        scores = self.classifier(x)
        return scores


class RAPN(nn.Module):
    def __init__(self, feature_dim, out_feature_dim, layer_num=1, dropout_rate=0.7):
        super(RAPN, self).__init__()
        self.feature_dim = feature_dim
        self.out_feature_dim = out_feature_dim
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.out_feature_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
        # self.embedding = nn.Sequential(
        #     nn.Linear(feature_dim, self.out_feature_dim)
        # )
        self.p_classifier = Classifier(self.out_feature_dim, dropout_rate)
        self.sample_generator = Sample_generator()
        self.LPN = LPN(feature_dim=self.out_feature_dim)


    def forward(self, ref_nor, ref_abn, mode='normal', isTrain=True, gl=False):
        # print(ref_nor.shape)       # [bs, N, F]

        bs = int(ref_nor.shape[0])

        # ref_nor = self.aggr(ref_nor)
        ref_nor = ref_nor.permute(0, 2, 1)
        ref_nor = self.embedding(ref_nor)
        ref_nor = ref_nor.permute(0, 2, 1)

        # ref_abn = self.aggr(ref_abn)
        ref_abn = ref_abn.permute(0, 2, 1)
        ref_abn = self.embedding(ref_abn)
        ref_abn = ref_abn.permute(0, 2, 1)

        ref = torch.cat([ref_nor, ref_abn], dim=0)

        p_score = self.p_classifier(ref)[:, :, 0]
        p_score_nor = p_score[:bs]
        p_score_abn = p_score[bs:]

        if not isTrain:   # 测试时直接返回伪标签
            if not gl:   # gl的意思是，是否在测试时能够得到全部的video信息，按理说如果能够得到全部的video信息效果会更好
                         # 但在SHT实验发现也好不到哪里去，就不做这个实验了，因为也确实不符合实际，gl一直设为False
                return p_score_nor

        nor_feat_conf_nor, nor_score_conf_nor, nor_feat_hard, nor_idx_hard \
            = self.sample_generator(ref_nor, p_score_nor, type='Normal')
        abn_feat_conf_nor, abn_score_conf_nor, abn_idx_conf_nor, abn_feat_conf_abn, abn_score_conf_abn, abn_feat_hard, abn_feat_score, abn_idx_hard \
            = self.sample_generator(ref_abn, p_score_abn, type='Abnormal')

        abn_score_hard, abn_score_topK_nor, abn_score_topK_abn, nor_new_feat_nor, new_feat_nor, new_feat_abn = \
            self.LPN(nor_feat_conf_nor, abn_feat_conf_nor, abn_score_conf_nor, abn_feat_conf_abn, abn_score_conf_abn, abn_feat_hard, abn_feat_score, type='Abnormal')

        # 给这一批样本中的hard样本打上新的标签
        nor_label = torch.zeros([bs, ref_nor.shape[1]]).cuda()
        abn_label = torch.ones([bs, ref_abn.shape[1]]).cuda()
        abn_idx_hard = [
            torch.arange(0, abn_idx_hard.shape[0]).view(-1, 1).repeat(1, abn_idx_hard.shape[1]).view(-1),
            abn_idx_hard.view(-1)]
        # 将hard example的标签硬化
        # abn_score_hard = torch.where(abn_score_hard > 0.8, 1., 0.)   # 0.8
        abn_label[abn_idx_hard] = abn_score_hard.view(-1)

        # 本来想给这批样本的异常视频中的置信nor样本打上0的标签，但实验发现这样在SHT上会发散，因为确实只靠伪标签并不能确定他就是nor的
        # 但在UCF上表现更好
        abn_idx_conf_nor = [
            torch.arange(0, abn_idx_conf_nor.shape[0]).view(-1, 1).repeat(1, abn_idx_conf_nor.shape[1]).view(-1),
            abn_idx_conf_nor.view(-1)]
        # abn_label[abn_idx_conf_nor] = torch.zeros_like(abn_score_conf_nor).view(-1)

        label = torch.cat([nor_label, abn_label], dim=0)

        # p_score = torch.mean(p_score.view(-1, 10), dim=1)
        # label = torch.mean(label.view(-1, 10), dim=1)

        if not isTrain:
            if not gl:   # gl的意思是，是否在测试时能够得到全部的video信息，按理说如果能够得到全部的video信息效果会更好
                         # 但在SHT实验发现也好不到哪里去，就不做这个实验了，因为也确实不符合实际，gl一直设为False
                return p_score_nor
            else:
                p_ori_score = p_score_nor
                video_label = 1 if torch.sum(abn_score_conf_abn) > 1.0 else 0
                train_utils.log('---------------')
                train_utils.log(video_label)

                train_utils.log(p_score_nor)

                idx_abn = torch.where((p_score_nor>0.) & (p_score_nor<1.7))
                ref_conf_abn = ref_nor[idx_abn].unsqueeze(1)
                # idx_nor = torch.where(p_score_nor < 0.2)
                # ref_conf_nor = ref_nor[idx_nor].unsqueeze(1)
                if ref_conf_abn.shape[0]!=0:
                    ref_aggr_score_abn =  self.LEN(ref_conf_abn, ref_conf_abn, ref_conf_abn, ref_conf_abn, isTrain=False).permute(1,0,2)[:,:,0]
                    train_utils.log('abn:')
                    train_utils.log(p_score_nor[idx_abn])
                    train_utils.log(ref_aggr_score_abn)
                    p_score_nor[idx_abn] = ref_aggr_score_abn.squeeze(0)


                # if ref_conf_nor.shape[0]!=0:
                    # ref_aggr_score_nor =  self.LEN(ref_conf_nor, ref_conf_nor, ref_conf_nor, ref_conf_nor, isTrain=False).permute(1,0,2)[:,:,0]
                    # train_utils.log('nor:')
                    # train_utils.log(p_score_nor[idx_nor])
                    # train_utils.log(ref_aggr_score_nor)
                    # p_score_nor[idx_nor] = ref_aggr_score_nor.squeeze(0)

                # if video_label == 0:
                #     return torch.zeros_like(p_score_nor)
                # print(ref_aggr_score.shape)
                return p_score_nor

        return p_score, label, abn_score_topK_nor, abn_score_topK_abn, nor_new_feat_nor, new_feat_nor, new_feat_abn

        # if not isTrain:
        #     print('============abnormal:')
        #     print(abn_score_conf_nor)
        #     print(abn_score_conf_abn)
        #     video_label = 1 if torch.sum(abn_score_conf_abn)>1.5 else 0
        #     print(video_label)
        #     if video_label == 1:
        #         abn_idx_hard = [
        #             torch.arange(0, abn_idx_hard.shape[0]).view(-1, 1).repeat(1, abn_idx_hard.shape[1]).view(-1),
        #             abn_idx_hard.view(-1)]
        #         abn_lpn_score_hard = abn_lpn_score[:, 10:]
        #         print(abn_lpn_score_hard)
        #         # p_score_abn[abn_idx_hard] = abn_lpn_score_hard.view(-1)
        #         return p_score_abn
        #     else:
        #         nor_idx_hard = [
        #             torch.arange(0, nor_idx_hard.shape[0]).view(-1, 1).repeat(1, nor_idx_hard.shape[1]).view(-1),
        #             nor_idx_hard.view(-1)]
        #         nor_lpn_score_hard = nor_lpn_label[:, 10:]
        #         print(nor_lpn_score_hard)
        #         p_score_nor[nor_idx_hard] = nor_lpn_score_hard.view(-1)
        #
        #         return p_score_nor


