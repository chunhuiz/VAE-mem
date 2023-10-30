# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 5.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig('1.jpg')
    return fig


def main():
    nor_keys = F.normalize(torch.rand((10, 2048), dtype=torch.float),
                           dim=1)
    abn_keys = F.normalize(torch.rand((10, 2048), dtype=torch.float),
                           dim=1)
    keys = torch.cat([nor_keys, abn_keys], 0)
    label = [0] * len(nor_keys) + [1] * len(abn_keys)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(keys.numpy())
    print('result.shape',result.shape)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    # plt.show(fig)


if __name__ == '__main__':
    main()
