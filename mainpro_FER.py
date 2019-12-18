'''Train Fer2013 with PyTorch.'''
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
from fer import FER2013
from torch.autograd import Variable
from models import *
import matplotlib.pyplot as plt
import visdom

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='Resnet18', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

# use_cuda = torch.cuda.is_available()
use_cuda = True

best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 20  # 50 学习率开始下降的epoch节点
learning_rate_decay_every = 2 # 5
learning_rate_decay_rate = 0.95 # 0.9

cut_size = 44
total_epoch = 100 # 250

path = os.path.join(opt.dataset + '_' + opt.model) # 存放结果的文件夹

# Prepare Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# 加载训练集、测试集
trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=0)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=0)



# 选择模型
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()


# 选择从 以前的checkpoint/未完成的model 继续的情况
if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss() # 交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4) # 优化器

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
public_test_loss_list = [0]
public_test_acc_list = [0]
private_test_loss_list = [0]
private_test_acc_list = [0]
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    print("Start Training!")
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # 学习率下降
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    # 迭代，迭代的次数 = 28708(训练集大小)/batch_size = batch数，计算loss和accuracy
    # batch_idx计迭代数
    iter_num = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs) # 正向传播
        loss = criterion(outputs, targets)
        loss.backward() # 反向传播
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        # train_loss += loss.data[0]
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0) # 每次加上batch_size个数
        correct += predicted.eq(targets.data).cpu().sum()

        Train_acc = int(correct) / int(total)  # 也就是最后一次迭代的准确率，代表一个epoch的准确率（完整的数据集通过了神经网络）

        # 输出loss和训练集准确率
        # utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        iter_num += 1


    train_loss = train_loss/iter_num
    train_loss_list.append(train_loss)
    train_acc_list.append(Train_acc)
    print("Train Accuracy:",Train_acc*100,"%")
    print("Train Loss:",train_loss)

def PublicTest(epoch):
    print("Start PublicTesting!")
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval() # eval()时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，
    # 不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大
    PublicTest_loss = 0
    correct = 0
    total = 0
    iter_num = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():  # add
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1) # 计算输出的平均值
        loss = criterion(outputs_avg, targets)
        # PublicTest_loss += loss.data[0]
        PublicTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        PublicTest_acc = int(correct) / int(total)  # 该epoch的public test准确率
        # 输出loss和PublicTest准确率
        # utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #                    % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        iter_num += 1


    PublicTest_loss = PublicTest_loss/iter_num
    public_test_loss_list.append(PublicTest_loss)
    public_test_acc_list.append(PublicTest_acc)
    print("PublicTest Accuracy:",PublicTest_acc*100,"%")
    print("PublicTest Loss:",PublicTest_loss)

    # 如果此次test结果好于历史最好成绩，则保存模型/checkpoints
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch):
    print("Start PrivateTesting!")
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    iter_num = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():  # add
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        # PrivateTest_loss += loss.data[0]
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        PrivateTest_acc = int(correct) / int(total)  # 该epoch的public test准确率
        # utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        iter_num += 1


    PrivateTest_loss = PrivateTest_loss/iter_num
    private_test_loss_list.append(PrivateTest_loss)
    private_test_acc_list.append(PrivateTest_acc)
    print("PrivateTest Accuracy:",PrivateTest_acc*100,"%")
    print("PrivateTest Loss:",PrivateTest_loss)

    # Save checkpoint.
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch


#开启visdom
vis1 = visdom.Visdom()
vis2 = visdom.Visdom()

# loss
win1 = vis1.line(
    X=np.array([0,0]),
    Y=np.array([0,0]),
    opts=dict(
        title='Loss on FER',
        showlegend=False,
        xtickmin=0,
        xtickmax=100,
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
        title='Accuracy on FER',
        showlegend=False,
        xtickmin=0,
        xtickmax=100,
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
        Y=np.array([public_test_loss_list[epoch], public_test_loss_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win1,
        update='new',
        name='PublicTest Loss '+str(epoch)
    )
    vis1.line(
        X=np.array([epoch, epoch + 1]),
        Y=np.array([private_test_loss_list[epoch], private_test_loss_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win1,
        update='new',
        name='PrivateTest Loss '+str(epoch)
    )

    vis2.line(
        X=np.array([epoch, epoch + 1]),
        Y=np.array([train_acc_list[epoch], train_acc_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win2,
        update='new',
        name='Train Acc '+str(epoch)
    )
    vis2.line(
        X=np.array([epoch, epoch + 1]),
        Y=np.array([public_test_acc_list[epoch], public_test_acc_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win2,
        update='new',
        name='PublicTest Acc ' + str(epoch)
    )
    vis2.line(
        X=np.array([epoch, epoch + 1]),
        Y=np.array([private_test_acc_list[epoch], private_test_acc_list[epoch + 1]]),
        opts=dict(markercolor=np.array([1]),
                  markersymbol='dot'),
        win=win2,
        update='new',
        name='PrivateTest Acc ' + str(epoch)
    )


# 一个epoch内把训练和测试全包了
for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PublicTest(epoch)
    PrivateTest(epoch)
    visualize(epoch)

drawCurve(train_loss_list,1,'Train Loss')
drawCurve(train_acc_list,1,'Train Accuracy')
drawCurve(public_test_loss_list,1,'PublicTest Loss')
drawCurve(public_test_acc_list,1,'PublicTest Accuracy')
drawCurve(private_test_loss_list,1,'PrivateTest Loss')
drawCurve(private_test_acc_list,1,'PrivateTest Accuracy')
plt.show()

print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
