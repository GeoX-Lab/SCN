import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from math import log
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding for CNN backbone"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=True)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_f(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_f, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AdaptiveAvgPool2d((1, 1))  # AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def resnet18_cbam_f(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_f(Bottleneck, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_f(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_f(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_f(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_f(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


class long_short_network(nn.Module):

    def __init__(self, long_model, method, task_size, feature_extractor=None, batch_norm=False):
        super(long_short_network, self).__init__()
        self.method = method
        self.batch_norm = batch_norm

        if self.method == 'gated' or self.method == 'matrix':
            self.transer_model = long_model
            self.transer_model.eval()

        if feature_extractor is None:
            self.short_model = resnet18_cbam_f(pretrained=False)
        else:
            self.short_model = feature_extractor

        if feature_extractor.name in ['resnet', 'res2net', 'pyconv', 'dcd']:
            self.short_model.fc = nn.Linear(2048, task_size, bias=True)
        else:
            raise NotImplementedError("fpn is not completed yet")

        if self.method == 'gated':
            # new layer's weights
            self.conv1_tg = conv1x1(64, 64, 1)  # create transfer gate embeding
            self.layer1_tg = conv1x1(256, 256, 1)
            # self.layer2_tg = conv1x1(128, 128, 1)
            # self.layer3_tg = conv1x1(256, 256, 1)
            # self.layer4_tg = conv1x1(512, 512, 1)
            self.layer2_tg = conv1x1(512, 512, 1)
            self.layer3_tg = conv1x1(1024, 1024, 1)
            self.layer4_tg = conv1x1(2048, 2048, 1)
            torch.nn.init.constant_(self.conv1_tg.weight, 1)
            torch.nn.init.constant_(self.layer1_tg.weight, 1)
            torch.nn.init.constant_(self.layer2_tg.weight, 1)
            torch.nn.init.constant_(self.layer3_tg.weight, 1)
            torch.nn.init.constant_(self.layer4_tg.weight, 1)
            torch.nn.init.constant_(self.conv1_tg.bias, 0)
            torch.nn.init.constant_(self.layer1_tg.bias, 0)
            torch.nn.init.constant_(self.layer2_tg.bias, 0)
            torch.nn.init.constant_(self.layer3_tg.bias, 0)
            torch.nn.init.constant_(self.layer4_tg.bias, 0)

            self.conv1_tp = conv1x1(64, 64, 1)  # create transfer projection embedding
            self.layer1_tp = conv1x1(256, 256, 1)
            # self.layer2_tp = conv1x1(128, 128, 1)
            # self.layer3_tp = conv1x1(256, 256, 1)
            # self.layer4_tp = conv1x1(512, 512, 1)
            self.layer2_tp = conv1x1(512, 512, 1)
            self.layer3_tp = conv1x1(1024, 1024, 1)
            self.layer4_tp = conv1x1(2048, 2048, 1)

            self.conv1_rg = conv1x1(64, 64, 1)  # create resnet gate embeding
            self.layer1_rg = conv1x1(256, 256, 1)
            # self.layer2_rg = conv1x1(128, 128, 1)
            # self.layer3_rg = conv1x1(256, 256, 1)
            # self.layer4_rg = conv1x1(512, 512, 1)
            self.layer2_rg = conv1x1(512, 512, 1)
            self.layer3_rg = conv1x1(1024, 1024, 1)
            self.layer4_rg = conv1x1(2048, 2048, 1)

            torch.nn.init.constant_(self.conv1_rg.weight, 1)
            torch.nn.init.constant_(self.layer1_rg.weight, 1)
            torch.nn.init.constant_(self.layer2_rg.weight, 1)
            torch.nn.init.constant_(self.layer3_rg.weight, 1)
            torch.nn.init.constant_(self.layer4_rg.weight, 1)
            torch.nn.init.constant_(self.conv1_rg.bias, 0)
            torch.nn.init.constant_(self.layer1_rg.bias, 0)
            torch.nn.init.constant_(self.layer2_rg.bias, 0)
            torch.nn.init.constant_(self.layer3_rg.bias, 0)
            torch.nn.init.constant_(self.layer4_rg.bias, 0)

            self.eca_conv1_pool, self.eca_conv1 = self.eca_layer(64)  # create eca layer
            self.eca_layer1_pool, self.eca_layer1 = self.eca_layer(256)
            self.eca_layer2_pool, self.eca_layer2 = self.eca_layer(512)
            self.eca_layer3_pool, self.eca_layer3 = self.eca_layer(1024)
            self.eca_layer4_pool, self.eca_layer4 = self.eca_layer(2048)

            # self.eca_layer2_pool, self.eca_layer2 = self.eca_layer(128)
            # self.eca_layer3_pool, self.eca_layer3 = self.eca_layer(256)
            # self.eca_layer4_pool, self.eca_layer4 = self.eca_layer(512)

            # # resnet50 block
            # self.res_conv1_conv = nn.Conv2d(256, 64, stride=1, kernel_size=1)
            # self.res_conv2_conv = nn.Conv2d(512, 256, stride=1, kernel_size=1)

            if batch_norm is True:
                self.conv1_bn = nn.BatchNorm2d(64)
                self.layer1_bn = nn.BatchNorm2d(256)
                # self.layer2_bn = nn.BatchNorm2d(128)
                # self.layer3_bn = nn.BatchNorm2d(256)
                # self.layer4_bn = nn.BatchNorm2d(512)
                self.layer2_bn = nn.BatchNorm2d(256)
                self.layer3_bn = nn.BatchNorm2d(512)
                self.layer4_bn = nn.BatchNorm2d(1024)

        elif self.method == 'matrix':
            self.conv1_tp = conv1x1(64, 64, 1)  # create transfer projection embeding
            self.layer1_tp = conv1x1(256, 256, 1)
            # self.layer2_tp = conv1x1(128, 128, 1)
            # self.layer3_tp = conv1x1(256, 256, 1)
            # self.layer4_tp = conv1x1(512, 512, 1)
            self.layer2_tp = conv1x1(512, 512, 1)
            self.layer3_tp = conv1x1(1024, 1024, 1)
            self.layer4_tp = conv1x1(2048, 2048, 1)

    def forward(self, x):
        features = []
        if self.method == 'gated' or self.method == 'matrix':
            with torch.no_grad():
                _, feature_list, _ = self.transer_model(x)
        x = self.short_model.conv1(x)
        x = self.short_model.bn1(x)
        x = self.short_model.relu(x)
        # x = self.maxpool(x)
        if self.method == 'gated':
            # transfer conv1
            g_conv1 = self.transfer_gate(self.conv1_tg, self.conv1_rg, x, feature_list[0].detach())
            x_long = self.eca_forward(feature_list[0].detach(), self.eca_conv1_pool, self.eca_conv1)
            x = self.transfer_out(self.conv1_tp, g_conv1, x, x_long)
            x = self.transfer_bn(x, 'conv1')
        elif self.method == 'metrix':
            x = self.conv1_tp(feature_list[0]) + x
        else:
            x = x
        features.append(x)

        x = self.short_model.layer1(x)
        if self.method == 'gated':
            # transfer layer1
            # only when resnet50
            g_layer1 = self.transfer_gate(self.layer1_tg, self.layer1_rg, x, feature_list[1].detach())
            x_long = self.eca_forward(feature_list[1].detach(), self.eca_layer1_pool, self.eca_layer1)
            x = self.transfer_out(self.layer1_tp, g_layer1, x, x_long)
            x = self.transfer_bn(x, 'layer1')
        elif self.method == 'matrix':
            x = self.layer1_tp(feature_list[1]) + x
        else:
            x = x
        features.append(x)

        x = self.short_model.layer2(x)
        if self.method == 'gated':
            # transfer layer2
            g_layer2 = self.transfer_gate(self.layer2_tg, self.layer2_rg, x, feature_list[2].detach())
            x_long = self.eca_forward(feature_list[2].detach(), self.eca_layer2_pool, self.eca_layer2)
            x = self.transfer_out(self.layer2_tp, g_layer2, x, x_long)
            x = self.transfer_bn(x, 'layer2')
        elif self.method == 'matrix':
            x = self.layer2_tp(feature_list[2]) + x
        else:
            x = x
        features.append(x)

        x = self.short_model.layer3(x)
        if self.method == 'gated':
            # transfer layer3
            g_layer3 = self.transfer_gate(self.layer3_tg, self.layer3_rg, x, feature_list[3].detach())
            x_long = self.eca_forward(feature_list[3].detach(), self.eca_layer3_pool, self.eca_layer3)
            x = self.transfer_out(self.layer3_tp, g_layer3, x, x_long)
            x = self.transfer_bn(x, 'layer3')
        elif self.method == 'matrix':
            x = self.layer3_tp(feature_list[3]) + x
        else:
            x = x
        features.append(x)

        x = self.short_model.layer4(x)
        if self.method == 'gated':
            # transfer layer4
            g_layer4 = self.transfer_gate(self.layer4_tg, self.layer4_rg, x, feature_list[4].detach())
            x_long = self.eca_forward(feature_list[4].detach(), self.eca_layer4_pool, self.eca_layer4)
            x = self.transfer_out(self.layer4_tp, g_layer4, x, x_long)
            x = self.transfer_bn(x, 'layer4')
        elif self.method == 'matrix':
            x = self.layer4_tp(feature_list[4]) + x
        else:
            x = x
        features.append(x)

        x = self.short_model.feature(x)
        x = x.view(x.size(0), -1)
        x = self.short_model.fc(x)

        return x, features

    # memory transfer gate
    def transfer_gate(self, T_g, W_g, f_short, f_long):
        f_long = torch.nn.functional.adaptive_avg_pool2d(f_long, (1, 1))
        f_short = torch.nn.functional.adaptive_avg_pool2d(f_short, (1, 1))
        a = T_g(f_long)
        b = W_g(f_short)
        return torch.sigmoid(a + b)

    # cal output of each layer
    def transfer_out(self, T_p, g, f_short, f_long, method=None):
        if method is None:
            return (1 - g) * F.relu(T_p(f_long)) + f_short

    # batch_norm for transfer
    def transfer_bn(self, x, layer):
        if self.batch_norm is True:
            if layer == 'conv1':
                x = self.conv1_bn(x)
            elif layer == 'layer1':
                x = self.layer1_bn(x)
            elif layer == 'layer2':
                x = self.layer2_bn(x)
            elif layer == 'layer3':
                x = self.layer3_bn(x)
            elif layer == 'layer4':
                x = self.layer4_bn(x)
        else:
            x = x
        return x

    def eca_layer(self, C, gamma=2, b=1):
        # x: input features with shape [N,C,H,W]
        # gamma, b: parameters of mapping function
        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        avg_pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2))
        return avg_pool, conv

    def eca_forward(self, x, avg_pool, conv):
        y = avg_pool(x)
        y = conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        return x * y.expand_as(x)
