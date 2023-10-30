import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_utils
import numpy as np
from model.modules_XD import Sample_generator, LPN


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
    def __init__(self, feature_dim, out_feature_dim, hard_thres, layer_num=1, dropout_rate=0.7):
        super(RAPN, self).__init__()
        self.feature_dim = feature_dim
        self.out_feature_dim = out_feature_dim
        self.hard_thres = hard_thres
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
        self.LPN = LPN(feature_dim=128)


        self.conv1d1 = nn.Conv1d(in_channels=feature_dim, out_channels=512, kernel_size=1, padding=0)
        self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, padding=0)
        self.approximator = nn.Sequential(nn.Conv1d(128, 64, 1, padding=0), nn.ReLU(),
                                          nn.Conv1d(64, 32, 1, padding=0), nn.ReLU())
        self.conv1d_approximator = nn.Conv1d(32, 1, 5, padding=0)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, ref_nor, ref_abn, mode='normal', isTrain=True, gl=False):
        # print(ref_nor.shape)       # [bs, N, F]

        bs = int(ref_nor.shape[0])
        #
        input_nor = ref_nor
        ref_nor = ref_nor.permute(0, 2, 1)  # for conv1d
        ref_nor = self.relu(self.conv1d1(ref_nor))
        ref_nor = self.dropout(ref_nor)
        ref_nor = self.relu(self.conv1d2(ref_nor))
        ref_nor_drop = self.dropout(ref_nor)

        score_nor = self.approximator(ref_nor_drop)
        score_nor = F.pad(score_nor, (4, 0))
        score_nor = self.conv1d_approximator(score_nor)
        score_nor = score_nor.permute(0, 2, 1).squeeze(2)   # [bs, T]
        score_nor = self.sigmoid(score_nor)

        input_abn = ref_abn
        ref_abn = ref_abn.permute(0, 2, 1)  # for conv1d
        ref_abn = self.relu(self.conv1d1(ref_abn))
        ref_abn = self.dropout(ref_abn)
        ref_abn = self.relu(self.conv1d2(ref_abn))
        ref_abn_drop = self.dropout(ref_abn)

        score_abn = self.approximator(ref_abn_drop)
        score_abn = F.pad(score_abn, (4, 0))
        score_abn = self.conv1d_approximator(score_abn)
        score_abn = score_abn.permute(0, 2, 1).squeeze(2)
        score_abn = self.sigmoid(score_abn)

        # ref_nor = ref_nor.permute(0, 2, 1)
        # ref_nor = self.embedding(ref_nor)
        # ref_nor = ref_nor.permute(0, 2, 1)
        #
        # # ref_abn = self.aggr(ref_abn)
        # ref_abn = ref_abn.permute(0, 2, 1)
        # ref_abn = self.embedding(ref_abn)
        # ref_abn = ref_abn.permute(0, 2, 1)
        #
        # ref = torch.cat([ref_nor, ref_abn], dim=0)
        #
        # p_score = self.p_classifier(ref)[:, :, 0]
        # score_nor = p_score[:bs]
        # score_abn = p_score[bs:]

        if not isTrain:
            return score_nor


        nor_feat_conf_nor, nor_score_conf_nor = self.sample_generator(ref_nor_drop.permute(0, 2, 1), score_nor, type='Normal')
        abn_feat_conf_nor, abn_score_conf_nor, abn_idx_conf_nor, abn_feat_conf_abn, abn_score_conf_abn, abn_idx_conf_abn, abn_feat_hard, abn_score_hard, abn_idx_hard \
            = self.sample_generator(ref_abn_drop.permute(0, 2, 1), score_abn, type='Abnormal')

        new_abn_score_hard, abn_score_topK_nor, abn_score_topK_abn, nor_new_feat_nor, new_feat_nor, new_feat_abn = \
            self.LPN(nor_feat_conf_nor, abn_feat_conf_nor, abn_score_conf_nor, abn_feat_conf_abn, abn_score_conf_abn,
                     abn_feat_hard, abn_score_hard, abn_idx_conf_nor, abn_idx_conf_abn, abn_idx_hard, type='Abnormal')

        nor_label = torch.zeros([bs, ref_nor.shape[2]]).cuda()
        abn_label = torch.ones([bs, ref_abn.shape[2]]).cuda()

        abn_idx_hard_list = [
            torch.arange(0, abn_idx_hard.shape[0]).view(-1, 1).repeat(1, abn_idx_hard.shape[1]).view(-1),
            abn_idx_hard.view(-1)]
        abn_idx_conf_nor_list = [
            torch.arange(0, abn_idx_conf_nor.shape[0]).view(-1, 1).repeat(1, abn_idx_conf_nor.shape[1]).view(-1),
            abn_idx_conf_nor.view(-1)]

        # abn_label = pl_abn.clone()
        abn_label[abn_idx_hard_list] = new_abn_score_hard.view(-1)
        # abn_label[abn_idx_conf_nor_list] = torch.zeros_like(abn_score_conf_nor).view(-1)
        # print('abn_label:')
        # print(abn_label)

        # abn_label = torch.where(abn_label > self.hard_thres, 1., 0.)

        abn_label_hard = torch.where(new_abn_score_hard > self.hard_thres, 1., 0.)


        return score_nor, score_abn, nor_label, abn_label, abn_score_hard, abn_label_hard, \
               abn_score_topK_nor, abn_score_topK_abn, nor_new_feat_nor, new_feat_nor, new_feat_abn


