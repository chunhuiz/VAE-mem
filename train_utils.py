import os
import shutil
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_log_path = None

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.vals=[]

    def __format__(self, format_spec):
        f=0
        if len(self.vals)!=0:
            f=(sum(self.vals)/len(self.vals))
        return ('{:'+format_spec+'}').format(f)

    def val(self):
        if len(self.vals) != 0:
            f = sum(self.vals) / len(self.vals)
        else:
            f=0
        return f

    def update(self,val):
        if isinstance(val,np.ndarray):
            self.vals.append(val[0])
        elif isinstance(val,np.float64):
            self.vals.append(val)
        else:
            self.vals.append(val.detach().cpu().item())


def log10(t):
    """
    Calculates the base-10 log of each element in t.
    @param t: The tensor from which to calculate the base-10 log.
    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = torch.log(t)
    denominator = torch.log(torch.FloatTensor([10.])).cuda()
    return numerator / denominator


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def set_save_path(path, remove=True):

    if os.path.exists(path):
        if remove and (input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
            os.makedirs(os.path.join(path, "models"))
    else:
        os.makedirs(path)
        os.makedirs(os.path.join(path, "models"))


def make_optimizer(params, name, args):

    if name == 'SGD':
        optimizer = optim.SGD(
            params,
            lr=args['lr'],
            momentum=args['mom'],
            weight_decay=args['weight_decay']
        )
    else:
        optimizer = optim.Adam(params, args['lr'], weight_decay=args['weight_decay'], betas=args['betas'])

    if args['lr_scheduler'] == 'StepLR':

        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args['step_size']),
                            gamma=args['gamma']
                        )

    elif args['lr_scheduler'] == 'MultiStepLR':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=args['step_size'],
                            gamma=args['gamma'],
                        )

    elif args['lr_scheduler'] == 'CosineAnnealingLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['T_max'])

    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def random_perturb(v_len, num_segments):
    """
    Given the length of video and sampling number, which segments should I choose?
    Random sampling is used.
    :param v_len: length of video
    :param num_segments: expected number of segments
    :return: a list of indices to sample
    """
    random_p = np.arange(num_segments) * v_len / num_segments
    for i in range(num_segments):
        if i < num_segments - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)