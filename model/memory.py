import math
import os
import torch
import torch.nn as nn
import train_utils
import numpy as np
from model.modules_mem import Sample_generator
import torch.nn.functional as F


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


class Memory(nn.Module):
    def __init__(self, feature_dim, out_feature_dim, layer_num=1, dropout_rate=0.7):
        super(Memory, self).__init__()
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
        self.p_classifier_mem = Classifier(self.out_feature_dim * 2, dropout_rate)
        self.sample_generator = Sample_generator()

    def get_score(self, feat, keys):
        if len(feat.size()) == 3:
            bs, N, d = feat.size()
            m, d = keys.size()

        score = torch.matmul(feat, torch.t(keys))  # b X N X m
        if len(feat.size()) == 3:
            score = score.view(bs * N, m)  # (b X N) X m

        score_query = F.softmax(score, dim=0) # (b X N) X m
        score_keys = F.softmax(score, dim=1) # (b X N) X m

        return score_query, score_keys

    def read(self, feat, keys):
        batch_size, N, dims = feat.size()  # [bs, N, F]

        softmax_score_feat, softmax_score_keys = self.get_score(feat, keys)

        feat_reshape = feat.contiguous().view(batch_size * N, dims)

        concat_memory = torch.matmul(softmax_score_keys.detach(), keys)  # (b X N) X d
        updated_query = torch.cat((feat_reshape, concat_memory), dim=1)  # (b X N) X 2d
        updated_query = updated_query.view(batch_size, N, 2 * dims)

        return updated_query, softmax_score_feat, softmax_score_keys

    def get_update_query(self, keys, max_indices, score, query, train):

        m, d = keys.size()
        if train:
            query_update = torch.zeros((m, d)).cuda()
            # random_update = torch.zeros((m,d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                else:
                    query_update[i] = 0

            return query_update
        else:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                else:
                    query_update[i] = 0

            return query_update

    def update(self, feat, keys, train):

        batch_size, N, dims = feat.size()  # b X h X w X d

        softmax_score_feat, softmax_score_keys = self.get_score(feat, keys)

        query_reshape = feat.contiguous().view(batch_size * N, dims)

        _, gathering_indices = torch.topk(softmax_score_keys, 1, dim=1)

        if train:
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_feat,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)

        else:
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_feat,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)

        return updated_memory.detach()

    def update_dynamic(self, feats, keys, epoch, thred, train):
        with open('1.txt', 'a') as f:
            f.write(str(len(keys)) + ' ')
        batch_size, N, dims = feats.size()
        feats = feats.contiguous().view(batch_size * N, dims)
        sim = torch.matmul(feats, torch.t(keys))
        sim_max, _ = torch.max(sim, 1)
        # print(sim_max)
        # print(keys.shape)
        # thred = 2
        # if epoch >=5 :
        #     thred = 2
        idx_keep = sim_max >= thred
        idx_extra = torch.where(sim_max < thred)[0][:1]
        feats_keep = feats[idx_keep]
        feats_extra = feats[idx_extra]

        softmax_score_feat, softmax_score_keys = self.get_score(feats_keep, keys)

        query_reshape = feats_keep

        _, gathering_indices = torch.topk(softmax_score_keys, 1, dim=1)

        mem_update = F.normalize(feats_extra, dim=1)
        if train:
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_feat,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)

        else:
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_feat,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        updated_memory = torch.cat([updated_memory, mem_update], dim=0)
        return updated_memory.detach()

    def gather_loss(self, query, keys, train):
        batch_size, N, dims = query.size()
        if train:

            loss = torch.nn.TripletMarginLoss(margin=1.0)
            loss_mse = torch.nn.MSELoss()
            softmax_score_query, softmax_score_memory = self.get_score(query, keys)

            query_reshape = query.contiguous().view(batch_size * N, dims)

            _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

            # 1st, 2nd closest memories
            pos = keys[gathering_indices[:, 0]]
            neg = keys[gathering_indices[:, 1]]
            top1_loss = loss_mse(query_reshape, pos.detach())
            gathering_loss = loss(query_reshape, pos.detach(), neg.detach())

            return gathering_loss, top1_loss


        else:
            loss_mse = torch.nn.MSELoss()

            softmax_score_query, softmax_score_memory = self.get_score(query, keys)

            query_reshape = query.contiguous().view(batch_size * N, dims)

            _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
            # 相似度越大，这个值就会越小
            gathering_loss = torch.mean((query_reshape - keys[gathering_indices].squeeze(1).detach()) ** 2, dim=1)
            # gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

            return gathering_loss

    def inter_mem_loss(self, nor_keys, abn_keys):
        sim = torch.matmul(nor_keys, torch.t(abn_keys))
        return torch.mean(sim)

    def forward(self, ref_nor, ref_abn, nor_keys, abn_keys, epoch, mode='normal', isTrain=True, gl=False):

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
            keys = torch.cat([nor_keys, abn_keys], dim=0)
            updated_feat, _, _ = self.read(ref_nor, keys)
            updated_p_score = self.p_classifier_mem(updated_feat)[:, :, 0]
            nor_compactness_loss = self.gather_loss(ref_nor, nor_keys, train=False)
            abn_compactness_loss = self.gather_loss(ref_nor, abn_keys, train=False)
            return p_score_nor, updated_p_score, nor_compactness_loss, abn_compactness_loss, ref_nor

        nor_feat_conf_nor, nor_score_conf_nor, nor_feat_hard, nor_score_hard \
            = self.sample_generator(ref_nor, p_score_nor, type='Normal')
        abn_feat_conf_nor, abn_score_conf_nor, abn_idx_conf_nor, abn_feat_conf_abn, abn_score_conf_abn, abn_feat_hard, abn_hard_score, abn_idx_hard \
            = self.sample_generator(ref_abn, p_score_abn, type='Abnormal')

        score_select = torch.cat([nor_score_hard, abn_score_conf_abn], dim=0)

        nor_separateness_loss, nor_compactness_loss = self.gather_loss(nor_feat_conf_nor, nor_keys, train=True)
        abn_separateness_loss, abn_compactness_loss = self.gather_loss(abn_feat_conf_abn, abn_keys, train=True)

        separateness_loss = nor_separateness_loss + abn_separateness_loss
        compactness_loss = nor_compactness_loss + abn_compactness_loss
        inter_loss = self.inter_mem_loss(nor_keys, abn_keys)

        updated_nor_feat_hard, _, _ = self.read(nor_feat_hard, nor_keys)
        # print('nor')
        updated_nor_keys = self.update_dynamic(nor_feat_conf_nor, nor_keys, epoch, 4, train=True)

        updated_abn_feat_conf_abn, _, _ = self.read(abn_feat_conf_abn, abn_keys)
        # print('abnor')
        updated_abn_keys = self.update_dynamic(abn_feat_conf_abn, abn_keys, epoch, 4, train=True)

        updated_feat = torch.cat([updated_nor_feat_hard, updated_abn_feat_conf_abn], dim=0)
        updated_p_score = self.p_classifier_mem(updated_feat)[:, :, 0]

        nor_label = torch.zeros([bs, nor_score_hard.shape[1]]).cuda()
        abn_label = torch.ones([bs, abn_score_conf_abn.shape[1]]).cuda()
        label = torch.cat([nor_label, abn_label], dim=0)

        return score_select, updated_p_score, label, updated_nor_keys, updated_abn_keys, separateness_loss, compactness_loss, inter_loss
