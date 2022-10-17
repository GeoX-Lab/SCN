import itertools

import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch.nn import init
import math
from PIL import Image


# feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)  # last layer output put into current layer input
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


# define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.1)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


# plot feature map
def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))


# plot loss per task
def plot_loss_one_task(list, path):
    iters = range(len(list))

    plt.figure()

    plt.plot(iters, list, 'b', label='training loss')
    plt.title('Training loss')
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(path, dpi=300)
    # plt.show()


# plot accuracy per task
def plot_acc_one_task(list, path):
    iters = range(len(list))

    plt.figure()

    plt.plot(iters, list, 'g', label='test accuracy')
    plt.title('test accuracy')
    plt.xlabel('iters')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(path, dpi=300)
    # plt.show()


# define T-sigmoid function
def tempsigmoid(x, temp):
    return torch.sigmoid(x / (temp))


def plot_confusion_matrix(cm, task_num, class_num):
    ad = class_num - cm.shape[0]
    # print(ad)
    cm = np.pad(cm, (0, ad), 'constant')

    cmap = plt.cm.Blues

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.axis('off')

    ax = plt.gca()
    ax.set_facecolor('aliceblue')
    # left, right = plt.xlim()
    # ax.spines['left'].set_position(('data', left))
    # ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    plt.tight_layout()
    plt.savefig('result/%d.png' % task_num, transparent=True, dpi=800)

    plt.show()


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def draw_cam(model, img_path, save_path, transform=None, visheadmap=False):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0)
    model.eval()
    x = model.conv1(img)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    features = x  # 1x2048x7x7
    print(features.shape)
    output = model.avgpool(x)  # 1x2048x1x1
    print(output.shape)
    output = output.view(output.size(0), -1)
    print(output.shape)  # 1x2048
    output = model.fc(output)  # 1x1000
    print(output.shape)

    def extract(g):
        global feature_grad
        feature_grad = g

    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    features.register_hook(extract)
    pred_class.backward()
    greds = feature_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(2048):
        features[i, ...] *= pooled_grads[i, ...]
    headmap = features.detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)

    if visheadmap:
        plt.matshow(headmap)
        # plt.savefig(headmap, './headmap.png')
        plt.show()

    img = cv2.imread(img_path)
    headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
    headmap = np.uint8(255 * headmap)
    headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
    superimposed_img = headmap * 0.4 + img
    cv2.imwrite(save_path, superimposed_img)
