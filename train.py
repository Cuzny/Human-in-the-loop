import torch
import random
import logging
import numpy as np
import sys
import math
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.resnet18 import ResNet18
from models.hill_model import HillModel
from dataset import Cifar100Dataset
from argpar import get_args

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QLabel, QPushButton, QDialog, QRadioButton, QApplication
from PyQt5.QtGui import *

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='train.log',
                    filemode='w',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        self.init()
        ROW = (self.gallery_size + 4) // 5

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(750, (ROW + 1) * 140 + 20)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
                
        self.img = QtWidgets.QLabel(self)
        self.img.setGeometry(QtCore.QRect(40, 10, 100, 100))
        self.img.show()

        self.idx = QtWidgets.QLabel(self)
        self.idx.setGeometry(QtCore.QRect(40, 110, 200, 20))
        self.idx.show()

        self.imgs = []
        self.labels = []
        self.scores = []
        for row in range(ROW):
            for col in range(5):
                self.imgs.append(QLabel(self))
                self.imgs[row*5 + col].setGeometry(40 + col * 140, 150 + row * 140, 90, 90)
                self.imgs[row*5 + col].show()

                self.labels.append(QPushButton(self))
                self.labels[row*5 + col].setGeometry(40 + col * 140, 240 + row * 140, 90, 25)
                self.labels[row*5 + col].clicked.connect(self.btn_choose)
                self.labels[row*5 + col].show()

                self.scores.append(QLabel(self))
                self.scores[row*5 + col].setGeometry(40 + col * 140, 265 + row * 140, 90, 25)
                self.scores[row*5 + col].show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

    #按钮响应事件
    def btn_choose(self):
        text = self.sender().text()
        select_num = int(text.split(':')[1]) 
        #pround为当前进行到第多少张图
        print('epoch:' + str(self.epoch) + ', pround:' + str(self.p) + ', select class:' + str(select_num))
        self.hill_model.humanSelectPositive(self.probe_fea, select_num)
        self.count_and_eval()
        self.train()

    #程序计数
    def count_and_eval(self):
        self.p += 1
        #每100张图像验证一次准确率
        if self.p % self.et == 0:
            self.evaluate_train()
            self.evaluate_eval()
        if self.p == self.psize:
            self.p = 0
            self.epoch += 1
        if self.epoch == self.max_epoch:           
            print('Saving model to models/hill_model_final.pkl...')
            torch.save(self.hill_model.state_dict(), 'models/hill_model_final.pkl')
            sys.exit(app.exec_())

    #数据初始化
    def init(self):

        self.args = get_args()

        # set torch random seed
        torch.manual_seed(2345)
        torch.cuda.manual_seed_all(2345)
        torch.backends.cudnn.deterministic = True

        # set cuda
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(0)
            print(torch.cuda.current_device())
            print(torch.cuda.get_device_name(torch.cuda.current_device()))

        # set dataset
        train_folder_path = './datasets/cifar100/train_10/'
        val_folder_path = './datasets/cifar100/val_10/'
        self.train_class_num = self.args.train_class_num
        self.train_dataset = Cifar100Dataset(train_folder_path, self.train_class_num, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch, shuffle=True)
        logging.info('train_set length : {}'.format(len(self.train_dataset)))
        self.gallery_seen_images = [self.train_dataset.imgs_seen[i][0] for i in range(len(self.train_dataset.imgs))]
        self.gallery_labels = torch.arange(0, len(self.train_dataset.imgs))
        self.psize = len(self.train_dataset)

        self.val_dataset = Cifar100Dataset(val_folder_path, self.train_class_num, mode='eval')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.val_batch, shuffle=False)
        logging.info('val_set length : {}'.format(len(self.val_dataset)))

        # set model
        fea_weight_path = './models/resnet18.pkl'
        self.fea_model = ResNet18()
        self.fea_model.load_state_dict(torch.load(fea_weight_path))
        self.fea_model.eval()
        n_feature = 512
        self.hill_model = HillModel(n_feature=n_feature, train_gsize=self.train_class_num)
        self.hill_model.eval()      
        if self.use_cuda:
            self.fea_model = self.fea_model.cuda()
            self.hill_model = self.hill_model.cuda()
            
        # calculate all classes mean features
        imgs = self.train_dataset.imgs    # list(list(torch[3, 72, 72]))
        s_imgs = [torch.stack(imgs1class, dim=0).cuda() for imgs1class in imgs]    # list([num, 3, 72, 72])
        mean_features = []
        with torch.no_grad():
            for i, imgs1class in enumerate(s_imgs):
                features1class = self.fea_model(imgs1class)    # [num, 256]
                mean_feature1class = torch.mean(features1class, dim=0)
                mean_features.append(mean_feature1class)
        mean_features = torch.stack(mean_features, dim=0)
        self.hill_model.setClassFeatures(mean_features)

        #set params
        self.p = 0
        self.epoch = 0
        self.probe_label = -1
        self.probe_fea = []
        self.probe_seen_img = []
        self.best_acc = -1

        self.max_epoch = self.args.max_epoch
        self.et = self.args.eval_time
        self.gallery_size = self.args.gallery_size
        self.is_simu = self.args.is_simu


    #使用模型计算各个类的得分结果并返回得分，排名，以及当前图像所对应类的排名
    #用于训练阶段已知当前图像所对应的类
    def calculate_scores(self):
        probe_img, self.probe_seen_img, self.probe_label = self.train_dataset.load_train_data(self.p)
        img = probe_img.unsqueeze(0)
        if self.use_cuda:
            img = img.cuda()
        # forward
        self.probe_fea = self.fea_model(img)
        res, fsort_idx = self.hill_model.get_rank_list(self.probe_fea)
        if self.is_simu:
            g_rank = torch.nonzero(fsort_idx == self.probe_label, as_tuple=False)[0][0]
            return g_rank
        else:
            return res, fsort_idx

    #使用pyqt5显示排名前25的图像
    def show_images(self, res, fsort_idx):
        # imshow probe image
        self.idx.setText("probe image")
        # probe_seen_img为cv2.imread所读入数据
        img_src = self.probe_seen_img
        temp_imgSrc = QImage(img_src[:], img_src.shape[1], img_src.shape[0], img_src.shape[1] * 3, QImage.Format_BGR888)
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc)
        window.img.setPixmap(pixmap_imgSrc)

        # imshow gallery images
        for i in range(self.gallery_size):    
            label = self.gallery_labels[fsort_idx[i]].item()
            img_src = self.gallery_seen_images[fsort_idx[i]]
            temp_imgSrc = QImage(img_src[:], img_src.shape[1], img_src.shape[0], img_src.shape[1] * 3, QImage.Format_BGR888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc)
            window.imgs[i].setPixmap(pixmap_imgSrc)
            strT = 'sc: %.2f' %res[label].item()
            window.labels[i].setText("%s" %('pid:' + str(label)))
            window.scores[i].setText("%s" %(strT))

    #训练数据已知标签的模拟训练
    def train_simu(self):
        with torch.no_grad():         
            while True:
                g_rank = self.calculate_scores()
                #print('epoch:' + str(self.epoch) + ', pround:' + str(self.p) + ', iter:' + str(self.iter) + ', true rank:' + str(g_rank.item()))
                self.hill_model.humanSelectPositive(self.probe_fea, self.probe_label)
                self.count_and_eval()
                
    #训练函数，未知标签
    def train(self):
        with torch.no_grad():
            res, fsort_idx = self.calculate_scores()
            self.show_images(res, fsort_idx)

    #训练集上的准确率验证
    def evaluate_train(self):
        with torch.no_grad():
            train_acc = 0
            for x, y in tqdm(self.train_dataloader):
                '''
                batch = 1
                x : [batch, 3, 72, 72]
                y : [batch]
                '''
                batch_size = x.shape[0]
                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                x_feas = self.fea_model(x) # [1, n_fea]
                x_scores = self.hill_model(x_feas, y)    # [class_num]
                _, pred_x = x_scores.max(dim=1)
                num_correct_x = (pred_x == y).sum().item()
                acc = int(num_correct_x) / batch_size
                train_acc += acc
            avg_acc = train_acc/len(self.train_dataloader)
            logging.info('epoch: %d, pround: %d, train accuracies : %.4f' %(self.epoch, self.p, avg_acc))

    #验证集上的准确率验证
    def evaluate_eval(self):
        eval_acc = 0
        with torch.no_grad():
            for x, y in tqdm(self.val_dataloader):
                '''
                batch = 1
                x : [batch, 3, 72, 72]
                y : [batch]
                '''
                batch_size = x.shape[0]
                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                x_feas = self.fea_model(x) # [1, n_fea]
                x_scores = self.hill_model(x_feas, y)    # [class_num]

                _, pred_x = x_scores.max(dim=1)
                num_correct_x = (pred_x == y).sum().item()
                acc = int(num_correct_x) / batch_size
                eval_acc += acc
            
            avg_acc = eval_acc/len(self.val_dataloader)
            logging.info('epoch: %d, pround: %d, eval accuracies : %.4f' %(self.epoch, self.p, avg_acc))

            if avg_acc > self.best_acc:
                self.best_acc = avg_acc
                logging.info('Saving model to models/hill_model_best.pkl...')
                torch.save(self.hill_model.state_dict(), 'models/hill_model_best.pkl')       

#自己建一个mywindows类，mywindow是自己的类名。QtWidgets.QMainWindow：继承该类方法
class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    #__init__:析构函数，也就是类被创建后就会预先加载的项目。
    # 马上运行，这个方法可以用来对你的对象做一些你希望的初始化。
    def __init__(self):
        #这里需要重载一下mywindow，同时也包含了QtWidgets.QMainWindow的预加载项。
        super(mywindow, self).__init__()
        self.setupUi(self)

if __name__ == '__main__':

    # QApplication相当于main函数，也就是整个程序（很多文件）的主入口函数。
    # 对于GUI程序必须至少有一个这样的实例来让程序运行。
    app = QtWidgets.QApplication(sys.argv)
    #生成 mywindow 类的实例。
    window = mywindow()
    if window.is_simu:
        window.train_simu()
    else:
        window.show()
        window.train()
    sys.exit(app.exec_())