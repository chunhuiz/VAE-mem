import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

class Sample_generator(nn.Module):
    def __init__(self):
        super(Sample_generator, self).__init__()

    def idx_ano_generate(self, score, type='Abnormal'):
        if type == 'Abnormal':

            hard_topK = 10  # 挑选出video中的x个困难样本，即异常分数值接近于0.5
            idx_hard = torch.topk(torch.abs_(score - 0.5), hard_topK, dim=1, largest=False)[1]

            conf_nor_topK = 5  # 将video中置信度高的x个clip视为normal
            idx_conf_nor = torch.topk(score, conf_nor_topK, dim=1, largest=False)[1]

            conf_abn_topK = 10  # 挑选出video中置信度高的x个clip作为abnormal
            idx_conf_abn = torch.topk(score, conf_abn_topK, dim=1, largest=True)[1]

            return idx_conf_nor, idx_conf_abn, idx_hard

        else:
            hard_topK = 10  # 挑选出normal video中的x个困难样本，即误判为异常
            idx_hard = torch.topk(score, hard_topK, dim=1, largest=True)[1]

            conf_nor_topK = 10  # 将video中置信度高的x个clip视为normal
            idx_conf_nor = torch.topk(score, conf_nor_topK, dim=1, largest=False)[1]
            return idx_conf_nor, idx_hard

    def idx_sim_generate(self, feat):
        idx_hard = None
        return idx_hard

    def forward(self, feat, score, type='Abnormal'):
        # feat: [bs, T, F], scores: [bs, T]
        F_len = feat.shape[2]

        if type == 'Abnormal':
            idx_ano_conf_nor, idx_ano_conf_abn, idx_ano_hard = self.idx_ano_generate(score, type='Abnormal')
            # idx_sim_hard = self.idx_sim_generate(feat)
            # idx_hard = torch.cat([idx_ano_hard, idx_sim_hard], dim=1)
            idx_hard = idx_ano_hard
            feat_conf_nor = torch.gather(feat, 1, idx_ano_conf_nor.unsqueeze(2).expand([-1, -1, F_len]))
            score_conf_nor = torch.gather(score, 1, idx_ano_conf_nor)
            feat_conf_abn = torch.gather(feat, 1, idx_ano_conf_abn.unsqueeze(2).expand([-1, -1, F_len]))
            score_conf_abn = torch.gather(score, 1, idx_ano_conf_abn)
            feat_hard = torch.gather(feat, 1, idx_hard.unsqueeze(2).expand([-1, -1, F_len]))
            score_hard = torch.gather(score, 1, idx_hard)
            return feat_conf_nor, score_conf_nor, idx_ano_conf_nor, feat_conf_abn, score_conf_abn, feat_hard, score_hard, idx_hard

        else:
            idx_ano_conf_nor, idx_ano_hard = self.idx_ano_generate(score, type='Normal')
            feat_conf_nor = torch.gather(feat, 1, idx_ano_conf_nor.unsqueeze(2).expand([-1, -1, F_len]))
            score_conf_nor = torch.gather(score, 1, idx_ano_conf_nor)
            feat_hard = torch.gather(feat, 1, idx_ano_hard.unsqueeze(2).expand([-1, -1, F_len]))
            score_hard_nor = torch.gather(score, 1, idx_ano_hard)
            return feat_conf_nor, score_conf_nor, feat_hard, score_hard_nor

class LPN(nn.Module):
    def __init__(self, feature_dim):
        super(LPN, self).__init__()
        self.feature_dim = feature_dim
        self.fq = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim))
        self.fk = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim))

        # self.fq = nn.Sequential(
        #     nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
        #               stride=1, padding=1),
        #     nn.ReLU()
        # )
        # self.fk = nn.Sequential(
        #     nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
        #               stride=1, padding=1),
        #     nn.ReLU()
        # )

        self.Softmax = nn.Softmax(dim=-1)

        self.sigma = 1
        self.alpha = 0.5

    def forward(self, nor_feat_conf_nor, feat_conf_nor, score_conf_nor, feat_conf_abn, score_conf_abn, feat_hard, score_hard, type='Abnormal'):
        F_len = nor_feat_conf_nor.shape[-1]
        # [bs, T, F]
        # feat = torch.cat([feat_conf_nor, feat_conf_abn], dim=1).permute(0, 2, 1)
        # feat = self.fq(feat)
        # feat = feat.permute(0, 2, 1)
        #
        # feat_hard = feat_hard.permute(0, 2, 1)
        # feat_hard = self.fk(feat_hard)
        # feat_hard = feat_hard.permute(0, 2, 1)

        # feat = self.fq(torch.cat([nor_feat_conf_nor, feat_conf_nor, feat_conf_abn], dim=1))
        feat = self.fq(torch.cat([feat_conf_nor, feat_conf_abn], dim=1))

        feat_hard = self.fk(feat_hard)

        # score = torch.cat([torch.zeros(nor_feat_conf_nor.shape[0], nor_feat_conf_nor.shape[1]).cuda(),
        #                    (score_conf_nor+torch.zeros_like(score_conf_nor))/2, score_conf_abn], dim=1)
        score = torch.cat([torch.zeros_like(score_conf_nor), score_conf_abn], dim=1)

        sim = torch.matmul(feat_hard / (torch.norm(feat_hard, p=2, dim=2).unsqueeze(2)),
                           (feat / (torch.norm(feat, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))  # [bs, Nh, N]
        sim = self.Softmax(sim)
        new_score_hard = torch.matmul(sim, score.unsqueeze(2)).squeeze(2)   # [bs, Nh]
        score_total = torch.cat([score, new_score_hard], dim=1)

        feat_total = torch.cat([feat, feat_hard], dim=1)

        # print('conf nor score:')
        # print(score_conf_nor)
        # print('hard score:')
        # print(score_hard)
        # print('sim:')
        # print(sim[0])
        # print('new_score_hard')
        # print(new_score_hard)

        nor_topK_total = 10  # 10
        nor_topK = 5  # 3
        idx_nor = torch.topk(new_score_hard, nor_topK, dim=1, largest=False)[1]
        score_topK_nor = torch.gather(new_score_hard, 1, idx_nor)
        idx_nor_total = torch.topk(score_total, nor_topK_total, dim=1, largest=False)[1]
        feat_topK_nor = torch.gather(feat_total, 1, idx_nor_total.unsqueeze(2).expand([-1, -1, F_len]))

        abn_topK_total = 5   # 10
        abn_topK = 2  # 3
        idx_abn = torch.topk(new_score_hard, abn_topK, dim=1, largest=True)[1]
        score_topK_abn = torch.gather(new_score_hard, 1, idx_abn)
        idx_abn_total = torch.topk(score_total, abn_topK_total, dim=1, largest=True)[1]
        feat_topK_abn = torch.gather(feat_total, 1, idx_abn_total.unsqueeze(2).expand([-1, -1, F_len]))

        nor_feat_nor_sim = torch.matmul(nor_feat_conf_nor / (torch.norm(nor_feat_conf_nor, p=2, dim=2).unsqueeze(2)),
                                    (nor_feat_conf_nor / (torch.norm(nor_feat_conf_nor, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))  # [bs, Nh, N]
        nor_feat_nor_sim = self.Softmax(nor_feat_nor_sim)
        nor_new_feat_nor = torch.matmul(nor_feat_nor_sim, nor_feat_conf_nor)

        feat_nor_sim = torch.matmul(feat_topK_nor / (torch.norm(feat_topK_nor, p=2, dim=2).unsqueeze(2)),
                           (feat_topK_nor / (torch.norm(feat_topK_nor, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))  # [bs, Nh, N]
        feat_nor_sim = self.Softmax(feat_nor_sim)
        new_feat_nor = torch.matmul(feat_nor_sim, feat_topK_nor)

        feat_abn_sim = torch.matmul(feat_topK_abn / (torch.norm(feat_topK_abn, p=2, dim=2).unsqueeze(2)),
                                    (feat_topK_abn / (torch.norm(feat_topK_abn, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))  # [bs, Nh, N]
        feat_abn_sim = self.Softmax(feat_abn_sim)
        new_feat_abn = torch.matmul(feat_abn_sim, feat_topK_abn)

        # return new_score_hard, score_topK_nor, score_topK_abn, nor_feat_conf_nor, feat_topK_nor, feat_topK_abn

        return new_score_hard, score_topK_nor, score_topK_abn, nor_new_feat_nor, new_feat_nor, new_feat_abn

    def select_topK(self, sim):

        return sim


if __name__ == '__main__':
    input = torch.rand([32,8,2048], dtype=torch.float)
