'''
PyTorch version
Model: ResNet
DataSet: CiFar-100
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import os
import Globals


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / (torch.ger(w1, w2) + 1e-6)


class MarginCosineProduct_divide(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.1, requires_grad=True):
        super(MarginCosineProduct_divide, self).__init__()
        # print('MarginCosineProduct_divide')
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=requires_grad)
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")

    def normalize_weight(self, input, target):
        self.weight.data = F.normalize(self.weight.data, 2, 1)

    def forward(self, input, label):
        input = input.to(self.weight.dtype)  # the input should be changed as the weight

        cosine = cosine_sim(input, self.weight)

        if self.training:
            # mask = cosine > self.m
            one_hot = torch.zeros_like(cosine, dtype=torch.uint8)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            one_hot = (cosine > self.m) & one_hot
            output = self.s * cosine
            output[one_hot] -= self.s * self.m
            return output

        output = cosine
        return output


printInfo = True
weightInit = True
nodeCount = 0


def save(mat, saveDir, saveType):
    # saveDir
    path = './' + saveDir
    if not os.path.exists(path):
        os.mkdir(path)
    
    # epoch
    path = path + '/' + 'Epoch' + str(Globals.epoch)
    if not os.path.exists(path):
        os.mkdir(path)

    # iteration
    path = path + '/' + 'Iteration' + str(Globals.iteration)
    if not os.path.exists(path):
        os.mkdir(path)

    path = path + '/' + str(Globals.nodeCount) + '-' + saveType
    if saveDir == 'Forward':
        mat = mat.detach().cpu().numpy()
        mat = mat.reshape(mat.shape[0], -1)
        path = path + '_' + str(mat.shape[1]) + '_' + str(mat.shape[0]) + '.txt'
        with open(path, 'w') as f:
            batch_size, total = mat.shape
            # print(path)
            # print(batch_size, total)
            for i in range(batch_size):
                for j in range(total):
                    f.write(str(mat[i, j]))
                    f.write(' ')
                f.write('\n')
    else:
        mat = mat.detach().cpu().numpy()
        mat = mat.reshape(-1, mat.shape[-1])
        path = path + '_' + str(mat.shape[0]) + '_' + str(mat.shape[1]) + '.txt'
        with open(path, 'w') as f:
            row, col = mat.shape
            # print(path)
            # print(batch_size, total)
            for i in range(col):
                for j in range(row):
                    f.write(str(mat[j, i]))
                    f.write(' ')
                f.write('\n')


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # print('Conv shape:', self.conv1.weight.shape)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum=0.1, eps=0.001)
        nn.init.ones_(self.bn1.weight)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # print('Conv shape:', self.conv2.weight.shape)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.1, eps=0.001)
        nn.init.ones_(self.bn2.weight)

        self.shortcut = nn.Sequential()
        self.inc = False

        if stride != 1:
            self.inc = True
            self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes, momentum=0.1, eps=0.001)
            nn.init.ones_(self.bn3.weight)
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_planes, eps=0.001)
                self.conv3,
                self.bn3
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))

        if Globals.printInfo:
            Globals.nodeCount += 1
            save(x, 'Forward', 'convInput')

        out = self.conv1(x)

        if Globals.printInfo:
            save(out, 'Forward', 'convOutput')

        if Globals.printInfo:
            Globals.nodeCount += 1
            save(out, 'Forward', 'bnInput')

        out = self.bn1(out)

        if Globals.printInfo:
            save(out, 'Forward', 'bnOutput')

        out = F.relu(out)

        if Globals.printInfo:
            Globals.nodeCount += 1
            save(out, 'Forward', 'convInput')

        out = self.conv2(out)

        if Globals.printInfo:
            save(out, 'Forward', 'convOutput')

        if Globals.printInfo:
            Globals.nodeCount += 1
            save(out, 'Forward', 'bnInput')

        out = self.bn2(out)

        if Globals.printInfo:
            save(out, 'Forward', 'bnOutput')

        if self.inc:
            # for layer in self.shortcut:
            #     if printInfo and isFirstForward:
            #         nodeCount += 1
            #         save(x, 'convInput', nodeCount)
            #     x = layer(x)
            #     if printInfo and isFirstForward:
            #         save(x, 'convOutput', nodeCount)
            # out += x
            if Globals.printInfo:
                Globals.nodeCount += 2
            out += self.shortcut(x)
        else:
            out += x

        out = F.relu(out)
        return out


# total layers = 6 * numLayers + 2
class ResNet(nn.Module):
    cMap = [16, 32, 64]

    def __init__(self, num_layers, num_classes=10, fix_w=False):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, self.cMap[0], kernel_size=3, stride=1, padding=1, bias=False)
        # self.initialize(self.conv1, 'convWeight_cntk.txt')

        self.bn1 = nn.BatchNorm2d(self.cMap[0], momentum=0.1, eps=0.001)
        nn.init.ones_(self.bn1.weight)

        self.layer1 = self._make_layer(self.cMap[0], self.cMap[0], num_layers, 1)
        self.layer2 = self._make_layer(self.cMap[0], self.cMap[1], num_layers, 2)
        self.layer3 = self._make_layer(self.cMap[1], self.cMap[2], num_layers, 2)
        self.feature = nn.Linear(self.cMap[2], 512)
        nn.init.zeros_(self.feature.bias)

        self.fc = nn.Linear(512, num_classes, bias=False)
        # requires_grad = True
        # if fix_w:
        #     requires_grad = False
        # self.amsoftmax = MarginCosineProduct_divide(512, num_classes, s=10, m=0.1, requires_grad=requires_grad)
        # if fix_w:
        #     for p in self.fc.parameters():
        #         p.requires_grad = False

        # global printInfo, weightInit, nodeCount
        if Globals.printInfo:
            Globals.nodeCount = 0
        if Globals.weightInit:
            model = self.state_dict()
            for name, param in self.named_parameters():
                # initialize the weight of conv and fc layers from weight file
                if self.isConvWeight(name):
                    total = 1
                    for dimension in param.shape:
                        total *= dimension
                    row, col = param.shape[-1], total // param.shape[-1]
                    weight_path = './ConvWeight/ConvWeight' + '_' + str(row) + '_' + str(col) + '.txt'
                    if not os.path.exists(weight_path):
                        raise RuntimeError('No such file %s exists.' % weight_path)
                    weight = self.init_from_file(weight_path)
                    weight = weight.reshape(param.shape)
                    # print(name, weight.shape)
                    model[name] = weight
                elif self.isFCWeight(name):
                    row, col = param.shape
                    weight_path = './fcWeight/fcWeight' + '_' + str(row) + '_' + str(col) + '.txt'
                    if not os.path.exists(weight_path):
                        raise RuntimeError('No such file %s exists.' % weight_path)
                    weight = np.loadtxt(weight_path)
                    weight = np.transpose(weight)
                    weight = torch.from_numpy(weight).float()
                    # print(name, weight.shape)
                    model[name] = weight
            self.load_state_dict(model)


    @staticmethod
    def _make_layer(in_planes, out_planes, num_blocks, stride):
        layers = [BasicBlock(in_planes, out_planes, stride)]
        for i in range(num_blocks-1):
            layers.append(BasicBlock(out_planes, out_planes, 1))
        return nn.Sequential(*layers)

    def isConvWeight(self, name):
        return (name.split('.')[-2] == 'conv1' or name.split('.')[-2] == 'conv2' or
                name.split('.')[-2] == '0') and (name.split('.')[-1] == 'weight')

    def isFCWeight(self, name):
        return (name.split('.')[-2] == 'feature' or name.split('.')[-2] == 'fc') and \
               (name.split('.')[-1] == 'weight')

    @staticmethod
    def init_from_file(weight_path):
        arr = []
        with open(weight_path, 'r') as f:
            rows, cols = f.readline().split(' ')
            for line in f.readlines():
                temp = line.replace('\n', '').split(' ')
                for i in range(len(temp)):
                    temp[i] = float(temp[i])
                arr.append(temp)
            arr = np.array(arr)
            arr = torch.from_numpy(arr).float()
        return arr

    def forward(self, x, label=None):
        # print(self.bn1.weight)
        # print(self.bn1.bias)
        # print(x.shape)
        # print(self.conv1.weight.shape)
        if Globals.printInfo:
            Globals.nodeCount += 1
            save(x, 'Forward', 'convInput')
            # save(self.conv1.weight, 'Forward', 'convWeight', nodeCount)
        out = self.conv1(x)
        if Globals.printInfo:
            save(out, 'Forward', 'convOutput')
        # print(out.shape)
        # print(self.bn1.weight.shape)
        if Globals.printInfo:
            Globals.nodeCount += 1
            save(out, 'Forward', 'bnInput')
        out = self.bn1(out)
        if Globals.printInfo:
            save(out, 'Forward', 'bnOutput')
        # self.save(out, 'BNOutput_PyTorch.txt')
        # assert 1 > 2
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, kernel_size=8, stride=1)
        out = out.view(out.size(0), -1)

        if Globals.printInfo:
            Globals.nodeCount += 1
            save(out, 'Forward', 'fcInput')

        out = self.feature(out)

        if Globals.printInfo:
            save(out, 'Forward', 'fcOutput')

        if Globals.printInfo:
            Globals.nodeCount += 1
            save(out, 'Forward', 'fcInput')

        out = self.fc(out)

        if Globals.printInfo:
            save(out, 'Forward', 'fcOutput')

        # out = self.amsoftmax(out, label)

        return out


def ResNet20(fix_w=False):
    return ResNet(num_layers=3, fix_w=fix_w)


def ResNet110(fix_w=False):
    return ResNet(num_layers=18, fix_w=fix_w)


def Model(model_name='', fix_w=False):
    if model_name == 'ResNet20':
        return ResNet20(fix_w=fix_w)
    elif model_name == 'ResNet110':
        return ResNet110(fix_w=fix_w)
    else:
        raise RuntimeError('No such model %s.' % model_name)

