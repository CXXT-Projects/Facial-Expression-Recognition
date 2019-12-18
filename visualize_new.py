"""
测试图像并可视化结果
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# os.environ["CUDA_VISIBLE_DEVICE"] = "3"
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import time
import cv2


class detector:
    def __init__(self,in_path,out_path):
        self.in_path = in_path
        self.out_path = out_path

    def imread(self,in_path):
        """
        指定格式读取原图
        """
        raw_img_bgr = cv2.imread(in_path) # 读取BGR格式测试图片
        # plt.imshow(raw_img_bgr)
        # pylab.show()
        shape = raw_img_bgr.shape
        print("input size:",shape)

        return raw_img_bgr

    def bgr2gray(self,raw_img_bgr):
        """
        BGR转灰度
        """
        gray_img = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2GRAY)
        print(gray_img.shape)
        # plt.imshow(gray)
        # pylab.show()

        return gray_img

    def detect_face(self,gray_img):
        """
        抠出人脸部分
        """
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faceRects = classifier.detectMultiScale(gray_img)
        # 判断有无人脸
        if len(faceRects) > 0:
            return True
        else:
            return False

    def pick_face(self,gray_img):
        """
        抠出人脸部分
        """
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faceRects = classifier.detectMultiScale(gray_img)
        # 判断有无人脸
        if len(faceRects) > 0:
            print('detect face successfully!')
            for faceRect in faceRects:
                x, y, w, h = faceRect
            gray_img_cutout = gray_img[max(0, y - 50):y + h + 50, max(0, x - 50):x + w + 50]
            # plt.imshow(gray)
            # pylab.show()
            print(gray_img_cutout.shape)
        else:
            print('no face detected!')

        return gray_img_cutout

    def resize(self,gray_img_cutout):
        """
        resize成指定大小
        """
        gray_img_resize = cv2.resize(gray_img_cutout, (48, 48))
        print(gray_img_resize.shape)
        # plt.imshow(gray)
        # pylab.show()

        return gray_img_resize

    def add_channel(self,gray_img_resize):
        """
        单通道转三通道
        """
        img = gray_img_resize[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        plt.imshow(img)
        pylab.show()

        return img

    def img_10crop(self,img):
        """
        数据增强
        将图片在左上角，左下角，右上角，右下角，中心进行切割和并做镜像操作，这样的操作使得数据库扩大了10倍
        """
        cut_size = [44, 44]  # 44
        transform_test = transforms.Compose([
            transforms.TenCrop(cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        inputs = transform_test(img)

        return inputs

    def net_forward(self,inputs):
        """
        将这10张图片送入模型
        """
        net = VGG('VGG19')
        # net = ResNet18()
        checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
        # checkpoint = torch.load(os.path.join('FER2013_Resnet18\\model_saved', 'PrivateTest_model.t7'))
        # checkpoint = torch.load(os.path.join('CK+_Resnet18\\1', 'Test_model.t7'))
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        net.eval()

        ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        with torch.no_grad():  # add
            inputs = Variable(inputs)
            outputs = net(inputs)

        return ncrops,outputs

    def predict(self,ncrops,outputs):
        """
        将得到的概率取平均，最大的输出分类即为对应表情
        这种方法有效地降低了分类错误
        """
        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
        score = F.softmax(outputs_avg,dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)

        return score,predicted

    def plot(self,score,predicted):
        """
        可视化识别结果
        """
        plt.rcParams['figure.figsize'] = (13.5,5.5)
        # 左图
        axes=plt.subplot(1, 3, 1)
        raw_img_rgb = io.imread(self.in_path) # 读取RGB格式测试图片
        plt.imshow(raw_img_rgb)
        plt.xlabel('Input Image', fontsize=16)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
        # 中间图
        plt.subplot(1, 3, 2)
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # class_names = ['anger','contempt','disgust','fear','happy','sadness','surprise']
        ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
        width = 0.4       # the width of the bars: can also be len(x) sequence
        color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
        for i in range(len(class_names)):
            plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
        plt.title("Classification results ",fontsize=20)
        plt.xlabel(" Expression Category ",fontsize=16)
        plt.ylabel(" Classification Score ",fontsize=16)
        plt.xticks(ind, class_names, rotation=45, fontsize=14)
        # 右图
        axes=plt.subplot(1, 3, 3)
        emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
        # emojis_img = io.imread('images/CK+_emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
        plt.imshow(emojis_img)
        plt.xlabel('Emoji Expression', fontsize=16)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.tight_layout()

        #plt.show()
        plt.savefig(os.path.join(self.out_path))
        plt.close()

        print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))

    def plot_noface(self):
        """
        可视化识别结果
        """
        plt.rcParams['figure.figsize'] = (9,5.5)
        # 左图
        axes=plt.subplot(1, 2, 1)
        raw_img_rgb = io.imread(self.in_path) # 读取RGB格式测试图片
        plt.imshow(raw_img_rgb)
        plt.xlabel('Input Image', fontsize=16)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
        # 右图
        axes=plt.subplot(1, 2, 2)
        emojis_img = io.imread('images/emojis/No Face.jpg')
        # emojis_img = io.imread('images/CK+_emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
        plt.imshow(emojis_img)
        plt.xlabel('No Face Detected!', fontsize=16)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.tight_layout()

        #plt.show()
        plt.savefig(os.path.join(self.out_path))
        plt.close()

        print("No Face Detected!")

def detect(in_path,out_path):
    """
    实现detector
    """
    det = detector(in_path,out_path)
    raw_img_bgr = det.imread(in_path)
    gray_img = det.bgr2gray(raw_img_bgr)
    if(det.detect_face(gray_img)):
        gray_img_cutout = det.pick_face(gray_img)
        gray_img_resize = det.resize(gray_img_cutout)
        img = det.add_channel(gray_img_resize)
        inputs = det.img_10crop(img)
        ncrops,outputs = det.net_forward(inputs)
        score,predicted = det.predict(ncrops,outputs)
        det.plot(score,predicted)
    else:
        det.plot_noface()

# image number(from 1.jpg --> num.jpg, or set the range)
for i in range(1):
    start_time = time.time()
    print("No:",i+1)
    # input
    detect('images/test_images/'+str(i+1)+'.jpg','images/test_images/results/'+str(i+1)+'.png')
    end_time = time.time()
    total_time = end_time - start_time
    print("time:",total_time,"s\n")