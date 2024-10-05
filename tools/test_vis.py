import torch

import torch

from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pylab import *
from scipy.cluster.vq import *
import seaborn as sns
import numpy as np
import matplotlib.patheffects as PathEffects

import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.cm as cm

total_pred_label = []
total_target = []
total_feature = []

def scatter2(feat, label, epoch):
    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', 20))
    ax = plt.subplot(aspect='equal')
    list_plot=[]
    for i in range(11):
        if i == 0:
            continue
        lala=feat[label == i]
        centroids, variance = kmeans(lala, 3)
        list_plot.append(centroids)
    for i in range(70, 80):
        lala = feat[label == i]
        centroids, variance = kmeans(lala, 3)
        list_plot.append(centroids)
    #9:x,2
    # sc = ax.scatter(feat[:, 0], feat[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    for i in range(20):
        plt.plot(list_plot[i][:,0],list_plot[i][:,1],'.', c=palette[i])

    ax.axis('tight')
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
                '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'
                ], loc='upper right')
    for i in range(20):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
    plt.draw()
    plt.savefig('/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/token_3_bs16_spilt1_gong_256/centerloss_{}.png'.format(epoch))
    plt.pause(0.001)


def scatter(feat, label, epoch):
    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', 4))
    ax = plt.subplot(aspect='equal')
    list_plot=[]

    # sc = ax.scatter(feat[:, 0], feat[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    for i in range(3):
        if i==0:
            continue
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i-1])
    for i in  range(70,72):
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i-68])
    plt.legend(['0', '1', '2', '3'], loc='upper right')
    ax.axis('tight')
    for i in range(2):
        if i == 0:
            continue
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i-1), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
    for i in  range(70,72):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i - 68), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
    plt.draw()
    plt.savefig('/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/token_3_bs16_spilt1_gong_256/centerloss_{}.png'.format(epoch))
    plt.pause(0.001)


with open('/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/token_3_bs16_spilt1_gong_256/test/test.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        strr=line.split(' ')
        total_target.append(int(strr[4]))
        total_feature.append([float(strr[0]),float(strr[2])])

a=torch.tensor([34,56,46,546,4645,64,456])
b=torch.tensor([34])
c=torch.stack((a.unsqueeze(0),b.unsqueeze(0)),dim=1)
print(c)
scatter( np.array(total_feature),np.array(total_target), 210)
