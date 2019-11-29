import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Model, save
import argparse
from configparser import ConfigParser
import numpy as np
import logging
import os
import cv2
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import Globals


def img2txt(image, path):
    with open(path, 'w') as f:
        for i in range(3):
            for j in range(32):
                for k in range(32):
                    f.write(str(image[i, j, k]))
                    f.write(' ')
        f.write('\n')


def rgb2bgr(image):
    image = image.numpy()
    image = np.flip(image, 0).copy()
    # img2txt(image, 'convInput_pytorch.txt')
    image = torch.from_numpy(image)
    return image


def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = 0
    logging.info('==> Starting Epoch %d' % (epoch+1))
    if Globals.printInfo:
        Globals.epoch = epoch + 1
    for batch_idx, (inputs, labels) in enumerate(CIFAR100_train_loader):
        for i in range(len(inputs)):
            inputs[i] = rgb2bgr(inputs[i])
        inputs, labels = inputs.to(device), labels.to(device)
        iteration += 1
        
        if Globals.printInfo:
            Globals.iteration = iteration
            Globals.nodeCount = 0

        optimizer.zero_grad()
        outputs = model(inputs, labels)
        loss = criterion(outputs, labels)
        
        # print('Iter %d: loss = %.8f' % (iteration, loss))

        loss.backward()

        # count = 0
        # for p in model.parameters():
        #     count += 1
            # print(p.requires_grad, p.grad.data.shape)
            # save(p.grad.data, 'Backward', 'Grad', count)

        optimizer.step()

        batch_size = labels.size(0)
        train_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        batch_correct = predicted.eq(labels).sum().item()
        correct += batch_correct

        logging.info('Training Epoch %d: [Iteration %d, loss: %.8f * %d | acc: %.6f%%]' % (
            epoch + 1, iteration, loss.item(), batch_size, batch_correct * 100 / batch_size))

        assert 1 > 2
    logging.info('Finished Epoch %d, loss: %.6f | acc: %.6f%%' % (epoch+1, train_loss/total, correct*100/total))


def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    logging.info('==> Test')
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(CIFAR100_test_loader):
            for i in range(len(inputs)):
                inputs[i] = rgb2bgr(inputs[i])
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            test_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    logging.info('Epoch %d, loss: %.6f | acc: %.6f%%' % (epoch+1, test_loss/total, correct*100/total))

    acc = correct*100/total
    if acc > best_acc:
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, model_path)
        best_acc = acc


def adjust_learning_rate():
    # if epoch == 1:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.1
    # elif epoch == 81:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.01
    # elif epoch == 121:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.001
    if epoch == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1
    elif epoch == 81:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
    elif epoch == 121:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--path', '--p', default='', type=str, help='config file')
    args = parser.parse_args()
    config_path = args.path
    if config_path == '':
        raise RuntimeError('configfile not exist.')

    config = ConfigParser()
    config.read(config_path)
    log_path = config.get('Model', 'Log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info('==> Loading config file...')
    for section in config.sections():
        logging.info(config.items(section))

    model_name = config.get('Model', 'ModelName')
    model_path = config.get('Model', 'ModelPath')
    model_path = os.path.join('checkpoint', model_path)
    resume = config.getboolean('Model', 'Resume')
    train_batch_size = config.getint('Train', 'train_batch_size')
    test_batch_size = config.getint('Test', 'test_batch_size')
    maxEpochs = config.getint('Train', 'maxEpochs')
    momentum = config.getfloat('Train', 'momentum')
    weight_decay = config.getfloat('Train', 'weight_decay')
    train_shuffle = config.getboolean('Train', 'train_shuffle')
    test_shuffle = config.getboolean('Test', 'test_shuffle')
    fix_w = config.getboolean('Train', 'fix_w')
    weightFile = config.get('Train', 'weightFile')

    # torch.set_printoptions(precision=10)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor()
        # transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        #                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        #                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    train_set = datasets.CIFAR10(root='./train_set', train=True, transform=transform_train,
                                  target_transform=None, download=False)

    # for i in range(len(train_set)):
    #     image, label = train_set[i]
    #     image = np.array(image)
    #     image = image.transpose([2, 0, 1])
    #     image = torch.from_numpy(image)
    #     image = rgb2bgr(image)
    #     logging.info('==> Building model...')
    #     model = Model(model_name=model_name)
    #     raise RuntimeError('Stop here.')

    logging.info(train_set)  # print the details of train_set
    test_set = datasets.CIFAR10(root='./test_set', train=False, transform=transform_test,
                                 target_transform=None, download=False)
    logging.info(test_set)  # print the details of test_set
    CIFAR100_train_loader = DataLoader(train_set, shuffle=train_shuffle, num_workers=2, batch_size=train_batch_size)
    CIFAR100_test_loader = DataLoader(test_set, shuffle=test_shuffle, num_workers=2, batch_size=test_batch_size)

    logging.info('==> Building model...')
    model = Model(model_name=model_name, fix_w=fix_w)
    model = model.to(device)

    # for n, p in model.named_parameters():
    #     print(n)

    if fix_w:
        if not os.path.exists(weightFile):
            raise RuntimeError('Weight file not exists.')
        weight = np.loadtxt(weightFile)
        weight = torch.tensor(weight)
        model.fc.weight.data.copy_(weight)  # load weight
        # model.amsoftmax.weight.data.copy_(weight)
        logging.info('==> Loading weight file %s.' % weightFile)

    logging.info(model)
    # summary(model, (3, 32, 32))  # print the details of model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum, weight_decay=weight_decay)
    start_epoch, end_epoch = 0, maxEpochs
    best_acc = 0.0

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if resume:
        logging.info('==> Resuming from checkpoint...')
        if not os.path.exists(model_path):
            raise RuntimeError('Error: no checkpoint file found.')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # writer = SummaryWriter()

    logging.info('==> Start training...')
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    for epoch in range(start_epoch, end_epoch):
        # scheduler.step(epoch)
        adjust_learning_rate()
        train()
        test()
    logging.info('Best Accuracy: %.6f%%' % best_acc)
