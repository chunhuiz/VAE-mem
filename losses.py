import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Total_loss2(torch.nn.Module):
    def __init__(self, config):
        super(Total_loss2, self).__init__()
        self.feat_alpha = config['loss_feat_alpha']
        self.score_alpha = config['loss_score_alpha']

        self.margin = config['loss_margin']
        self.loss_sparse = config['loss_sparse']
        self.loss_smooth = config['loss_smooth']

        # self.criterion = torch.nn.BCELoss()
        self.criterion = Weighted_BCE_Loss(weights=config['class_reweights'], label_smoothing=config['label_smoothing'])

    def forward(self, score_nor, score_abn, nor_label, abn_label, abn_score_hard, abn_label_hard,
                abn_score_topK_nor, abn_score_topK_abn, nor_feat_conf_nor, abn_feat_conf_nor, abn_feat_conf_abn, hard_label=True):
        print('abn_score_hard')
        print(abn_score_hard)
        print('abn_label_hard')
        print(abn_label_hard)
        if not hard_label:
            score_nor_idx = torch.topk(score_nor, k=int(score_nor.shape[1] // 16 + 1), largest=True)[1]
            score_nor_topK = torch.gather(score_nor, 1, score_nor_idx)
            # nor_label_topK = torch.gather(nor_label, 1, score_nor_idx)
            nor_label_topK = torch.zeros_like(score_nor_topK)

            score_abn_idx = torch.topk(score_abn, k=int(score_abn.shape[1] // 16 + 1), largest=True)[1]
            score_abn_topK = torch.gather(score_abn, 1, score_abn_idx)
            # abn_label_topK = torch.gather(abn_label, 1, score_abn_idx)
            abn_label_topK = torch.ones_like(score_abn_topK)
            loss_cls = self.criterion(score_nor_topK, nor_label_topK) + self.criterion(score_abn_topK, abn_label_topK)

        else:
            score_nor_idx = torch.topk(score_nor, k=int(score_nor.shape[1] // 16 + 1), largest=True)[1]
            score_nor_topK = torch.gather(score_nor, 1, score_nor_idx)
            # nor_label_topK = torch.gather(nor_label, 1, score_nor_idx)
            nor_label_topK = torch.zeros_like(score_nor_topK)

            score_abn_idx = torch.topk(score_abn, k=int(score_abn.shape[1] // 16 + 1), largest=True)[1]
            score_abn_topK = torch.gather(score_abn, 1, score_abn_idx)
            # abn_label_topK = torch.gather(abn_label, 1, score_abn_idx)
            abn_label_topK = torch.ones_like(score_abn_topK)

            # abn_score_hard = torch.gather(score_abn, 1, abn_idx_hard)
            # abn_label_hard = torch.gather(abn_label, 1, abn_idx_hard)

            loss_cls = self.criterion(score_nor_topK, nor_label_topK) + self.criterion(score_abn_topK, abn_label_topK) \
                       + self.criterion(abn_score_hard, abn_label_hard)

        score = torch.cat([score_nor, score_abn], dim=0)
        label = torch.cat([nor_label, abn_label], dim=0)

        loss_cls = self.criterion(score, label) + self.criterion(abn_score_hard, abn_label_hard)


        abn_score_topK_nor = torch.mean(abn_score_topK_nor)
        abn_score_topK_abn = torch.mean(abn_score_topK_abn)
        loss_abn_score_gap = F.relu(1.0 - torch.mean(abn_score_topK_abn - abn_score_topK_nor))
        print('loss_abn_score_gap')
        print(loss_abn_score_gap)

        nor_sim = torch.matmul(abn_feat_conf_nor / (torch.norm(abn_feat_conf_nor, p=2, dim=2).unsqueeze(2)),
                           (nor_feat_conf_nor / (torch.norm(nor_feat_conf_nor, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))

        abn_sim = torch.matmul(abn_feat_conf_nor / (torch.norm(abn_feat_conf_nor, p=2, dim=2).unsqueeze(2)),
                           (abn_feat_conf_abn / (torch.norm(abn_feat_conf_abn, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))

        loss_feat = torch.abs(self.margin - torch.mean(torch.mean(nor_sim, dim=2) - torch.mean(abn_sim, dim=2)))

        loss_total = loss_cls + self.feat_alpha * loss_feat + self.score_alpha * loss_abn_score_gap \
            + self.loss_sparse * sparsity(score_abn) + self.loss_smooth*smooth(score_abn)

        return loss_total

def CLAS2(logits, label):

    logits = logits.squeeze()
    k = int(logits.shape[1] // 16 + 1)
    logits = torch.topk(logits, k, dim=1, largest=True)[0]

    # logits = torch.sigmoid(logits)

    label = label.squeeze()
    label = label[:, :k]

    criterion = torch.nn.BCELoss()
    clsloss = criterion(logits, label)
    return clsloss

def CLAS(logits, label, seq_len, criterion, is_topk=True):
    logits = logits.squeeze()
    label = label.squeeze()
    instance_logits = torch.zeros(0).cuda()  # tensor([])
    for i in range(logits.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
            tmp = torch.mean(tmp).view(1)
        else:
            tmp = torch.mean(logits[i, :seq_len[i]]).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    instance_logits = torch.sigmoid(instance_logits)

    clsloss = criterion(instance_logits, label)
    return clsloss

class Mem_loss(torch.nn.Module):
    def __init__(self, config):
        super(Mem_loss, self).__init__()
        self.feat_alpha = config['loss_feat_alpha']
        self.score_alpha = config['loss_score_alpha']

        self.margin = config['loss_margin']
        self.loss_sparse = config['loss_sparse']
        self.loss_smooth = config['loss_smooth']

        # self.criterion = torch.nn.BCELoss()
        self.criterion = Weighted_BCE_Loss(weights=config['class_reweights'], label_smoothing=config['label_smoothing'])

    def forward(self, p_score, updated_p_score, label):
        #scores: [bs, T], labels:[bs, T], nor_feat_conf_nor:[bs, T1, F]
        bs = int(p_score.shape[0] / 2)
        p_scores_abn = p_score[bs:]
        loss_cls = 1.0*self.criterion(p_score, label)  # BCE loss in the score space
        update_loss_cls = 1.0 * self.criterion(updated_p_score, label)

        loss_total = loss_cls + update_loss_cls + self.loss_sparse * sparsity(p_scores_abn) + self.loss_smooth*smooth(p_scores_abn)

        return loss_total

class Total_loss(torch.nn.Module):
    def __init__(self, config):
        super(Total_loss, self).__init__()
        self.feat_alpha = config['loss_feat_alpha']
        self.score_alpha = config['loss_score_alpha']

        self.margin = config['loss_margin']
        self.loss_sparse = config['loss_sparse']
        self.loss_smooth = config['loss_smooth']

        # self.criterion = torch.nn.BCELoss()
        self.criterion = Weighted_BCE_Loss(weights=config['class_reweights'], label_smoothing=config['label_smoothing'])

    def forward(self, p_score, p_label, label, abn_score_topK_nor, abn_score_topK_abn, nor_feat_conf_nor, abn_feat_conf_nor, abn_feat_conf_abn):
        #scores: [bs, T], labels:[bs, T], nor_feat_conf_nor:[bs, T1, F]
        bs = int(p_score.shape[0] / 2)
        p_scores_abn = p_score[bs:]
        loss_cls = 1.0*self.criterion(p_score, label)  # BCE loss in the score space

        abn_score_topK_nor = torch.mean(abn_score_topK_nor)
        abn_score_topK_abn = torch.mean(abn_score_topK_abn)
        loss_abn_score_gap = F.relu(1.0 - torch.mean(abn_score_topK_abn - abn_score_topK_nor))

        nor_sim = torch.matmul(abn_feat_conf_nor / (torch.norm(abn_feat_conf_nor, p=2, dim=2).unsqueeze(2)),
                           (nor_feat_conf_nor / (torch.norm(nor_feat_conf_nor, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))

        abn_sim = torch.matmul(abn_feat_conf_nor / (torch.norm(abn_feat_conf_nor, p=2, dim=2).unsqueeze(2)),
                           (abn_feat_conf_abn / (torch.norm(abn_feat_conf_abn, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))

        loss_feat = torch.abs(self.margin - torch.mean(torch.mean(nor_sim, dim=2) - torch.mean(abn_sim, dim=2)))

        loss_total = loss_cls + self.feat_alpha * loss_feat + self.score_alpha * loss_abn_score_gap \
            + self.loss_sparse * sparsity(p_scores_abn) + self.loss_smooth*smooth(p_scores_abn)

        return loss_total

def sparsity(arr):
    # loss = torch.mean(torch.norm(arr, dim=0))
    loss = torch.mean(arr)

    return loss


def smooth(arr):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return loss


class Weighted_BCE_Loss(nn.Module):
    def __init__(self,weights,label_smoothing=0,eps=1e-8):
        super(Weighted_BCE_Loss, self).__init__()
        self.weights=weights
        self.eps=eps
        self.label_smoothing = label_smoothing
        # self.gamma=gamma

    def forward(self,scores,targets):
        # print(scores.shape)
        # print(targets.shape)
        new_targets=F.hardtanh(targets,self.label_smoothing,1-self.label_smoothing)
        return torch.mean(-self.weights[0]*new_targets*torch.log(scores+self.eps)\
                          -self.weights[1]*(1-new_targets)*torch.log(1-scores+self.eps))

class Flow_Loss(nn.Module):
    def __init__(self):
        super(Flow_Loss,self).__init__()

    def forward(self, gen_flows,gt_flows):

        return torch.mean(torch.abs(gen_flows - gt_flows))

class Intensity_Loss(nn.Module):
    def __init__(self,l_num):
        super(Intensity_Loss,self).__init__()
        self.l_num=l_num
    def forward(self, gen_frames,gt_frames):

        return torch.mean(torch.abs((gen_frames-gt_frames)**self.l_num))

class Gradient_Loss(nn.Module):
    def __init__(self,alpha,channels):
        super(Gradient_Loss,self).__init__()
        self.alpha=alpha
        filter=torch.FloatTensor([[-1.,1.]]).cuda()

        self.filter_x = filter.view(1,1,1,2).repeat(1,channels,1,1)
        self.filter_y = filter.view(1,1,2,1).repeat(1,channels,1,1)


    def forward(self, gen_frames,gt_frames):


        # pos=torch.from_numpy(np.identity(channels,dtype=np.float32))
        # neg=-1*pos
        # filter_x=torch.cat([neg,pos]).view(1,pos.shape[0],-1)
        # filter_y=torch.cat([pos.view(1,pos.shape[0],-1),neg.vew(1,neg.shape[0],-1)])
        gen_frames_x=nn.functional.pad(gen_frames,(1,0,0,0))
        gen_frames_y=nn.functional.pad(gen_frames,(0,0,1,0))
        gt_frames_x=nn.functional.pad(gt_frames,(1,0,0,0))
        gt_frames_y=nn.functional.pad(gt_frames,(0,0,1,0))

        gen_dx=nn.functional.conv2d(gen_frames_x,self.filter_x)
        gen_dy=nn.functional.conv2d(gen_frames_y,self.filter_y)
        gt_dx=nn.functional.conv2d(gt_frames_x,self.filter_x)
        gt_dy=nn.functional.conv2d(gt_frames_y,self.filter_y)

        grad_diff_x=torch.abs(gt_dx-gen_dx)
        grad_diff_y=torch.abs(gt_dy-gen_dy)

        return torch.mean(grad_diff_x**self.alpha+grad_diff_y**self.alpha)

class Adversarial_Loss(nn.Module):
    def __init__(self):
        super(Adversarial_Loss,self).__init__()
    def forward(self, fake_outputs):
        return torch.mean((fake_outputs-1)**2/2)
class Discriminate_Loss(nn.Module):
    def __init__(self):
        super(Discriminate_Loss,self).__init__()
    def forward(self,real_outputs,fake_outputs ):
        return torch.mean((real_outputs-1)**2/2)+torch.mean(fake_outputs**2/2)

class ObjectLoss(nn.Module):
    def __init__(self, device, l_num):
        super(ObjectLoss, self).__init__()
        self.device =device
        self.l_num=l_num

    def forward(self, outputs, target, flow, bboxes):
        # print(outputs.shape)
        # print(target.shape)
        # print(flow.shape)
        cof = torch.ones((outputs.shape[0], outputs.shape[2], outputs.shape[3])).to(self.device)
        boxcof = 2
        flowcof = 2
        for bbox in bboxes:
            cof[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] += boxcof

        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        rad = (u ** 2 + v ** 2)
        rad = rad.view(rad.shape[0], -1)

        min = rad.min(1)[0]
        max = rad.max(1)[0]

        rad = (rad - min) / (max - min)

        cof = torch.mul(cof, flowcof * (1+rad.view(rad.shape[0], flow.shape[-2], flow.shape[-1])))

        return torch.mean(torch.mul(cof, torch.abs((outputs - target) ** self.l_num)))