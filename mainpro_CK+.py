'''Train CK+ with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from CK import CK
from torch.autograd import Variable
from models import *
import matplotlib.pyplot as plt
import visdom
import time

parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
parser.add_argument('--model', type=str, default='Resnet18', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='CK+', help='dataset')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--bs', default=128, type=int, help='batch_size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

# use_cuda = torch.cuda.is_available()
use_cuda = True

best_Test_acc = 0  # best Test accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 20  # 50
learning_rate_decay_every = 1 # 5
learning_rate_decay_rate = 0.9 # 0.8

cut_size = 44
total_epoch = 60 # 60

path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# 数据扩增
# 随机切割，扩充数据集，减缓了过拟合的作用
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = CK(split = 'Training', fold = opt.fold, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
testset = CK(split = 'Testing', fold = opt.fold, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=0)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model == 'Resnet18':
    net = ResNet18()

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'Test_model.t7'))
    
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0)

batch_size = trainloader.batch_size
print("batch size =",batch_size)
print("Net =",opt.model)

# interval: 隔几个epoch取一个点
def drawCurve(in_list,interval,title):
    plt.figure()
    t = np.arange(len(in_list))
    plt.style.use("ggplot")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(interval*t, in_list) # 确保epoch的显示不受interval影响


train_loss_list = [0]
train_acc_list = [0]
test_loss_list = [0]
test_acc_list = [0]
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    print("Start Training!")
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))


    # 迭代，迭代的次数 = 981(训练集大小)/batch_size = batch数，计算loss和accuracy
    # batch_idx计迭代数
    iter_num = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        # train_loss += loss.data[0]
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        Train_acc = int(correct) / int(total)  # 也就是最后一次迭代的准确率，代表一个epoch的准确率（完整的数据集通过了神经网络）

        # utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        iter_num += 1

    train_loss = train_loss / iter_num
    train_loss_list.append(train_loss)
    train_acc_list.append(Train_acc)
    print("Train Accuracy:",Train_acc*100,"%")
    print("Train Loss:",train_loss)

def test(epoch):
    print("Start Testing!")
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    iter_num = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():  # add
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

        loss = criterion(outputs_avg, targets)
        # test_loss += loss.data[0]
        test_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        Test_acc = int(correct) / int(total)

        # utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        iter_num += 1

    test_loss = test_loss/iter_num
    test_loss_list.append(test_loss)
    test_acc_list.append(Test_acc)
    print("Test Accuracy:",Test_acc*100,"%")
    print("Test Loss:",test_loss)


    # Save checkpoint.
    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {'net': net.state_dict() if use_cuda else net,
            'best_Test_acc': Test_acc,
            'best_Test_acc_epoch': epoch,
        }
        if not os.path.isdir(opt.dataset + '_' + opt.model):
            os.mkdir(opt.dataset + '_' + opt.model)
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'Test_model.t7'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch


#开启visdom
vis1 = visdom.Visdom()
vis2 = visdom.Visdom()

# loss
win1 = vis1.line(
    X=np.array([0, 0]),
    Y=np.array([0, 0]),
    opts=dict(
        title='Loss on CK+',
        showlegend=False,
        xtickmin=0,
        xtickmax=60,
        xtickstep=5,
        ytickmin=0,
        ytickmax=5,
        ytickstep=1,
        markersymbol='dot',
        markersize=5
    ),
    name='Loss'
)
# accuracy
win2 = vis2.line(
    X=np.array([0, 0]),
    Y=np.array([0, 0]),
    opts=dict(
        title='Accuracy on CK+',
        showlegend=False,
        xtickmin=0,
        xtickmax=60,
        xtickstep=5,
        ytickmin=0,
        ytickmax=1,
        ytickstep=0.1,
        markersymbol='dot',
        markersize=5
    ),
    name='Accuracy'
)

#visdom可视化函数
def visualize(epoch):
    vis1.line(
        X=np.array([epoch, epoch + 1]),
        Y=np.array([train_loss_list[epoch], train_loss_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win1,
        update='new',
        name='Train Loss '+str(epoch)
    )
    vis1.line(
        X=np.array([epoch, epoch + 1]),
        Y=np.array([test_loss_list[epoch], test_loss_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win1,
        update='new',
        name='Test Loss '+str(epoch)
    )

    vis2.line(
        X=np.array([epoch, epoch + 1]),
        Y=np.array([train_acc_list[epoch], train_acc_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win2,
        update='new',
        name='Train Accuracy '+str(epoch)
    )
    vis2.line(
        X=np.array([epoch, epoch + 1]),
        Y=np.array([test_acc_list[epoch], test_acc_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win2,
        update='new',
        name='Test Accuracy ' + str(epoch)
    )

start_time = time.time()
for epoch in range(start_epoch, total_epoch):
    train(epoch)
    test(epoch)
    visualize(epoch)

end_time = time.time()
total_time= end_time- start_time
print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)
print("time =",total_time,'s')

# drawCurve(train_loss_list,1,'Train Loss')
# drawCurve(train_acc_list,1,'Train Accuracy')
# drawCurve(test_loss_list,1,'Test Loss')
# drawCurve(test_acc_list,1,'Test Accuracy')
# plt.show()




