import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class Cifar100Dataset(Dataset):
    def __init__(self, folder_path, class_num, mode):
        self.folder_path = folder_path
        self.samples = []
        assert mode == 'train' or mode == 'eval'
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.img_paths, self.imgs, self.labels, self.imgs_seen = [], [], [], []
        for i in range(class_num):
            sub_folder_path = self.folder_path + self.num2name(i + 1) + '/'
            sub_img_paths = [img_path for img_path in os.listdir(sub_folder_path) if img_path.endswith('.jpg')]
            sub_imgs = [self.getImg(sub_folder_path + img_path) for img_path in sub_img_paths]
            sub_imgs_seen = [cv2.imread(sub_folder_path + img_path)  for img_path in sub_img_paths]
            self.img_paths.append(sub_img_paths)
            self.imgs.append(sub_imgs)
            self.imgs_seen.append(sub_imgs_seen)
            self.labels.append([i for m in range(len(sub_img_paths))])

        self.tot_imgs = [m for n in self.imgs for m in n]
        self.tot_labels = [m for n in self.labels for m in n]
        self.tot_imgs_seen = [m for n in self.imgs_seen for m in n]
        randnum = random.randint(0,100)
        random.seed(randnum)
        random.shuffle(self.tot_imgs)
        random.seed(randnum)
        random.shuffle(self.tot_labels)
        random.seed(randnum)
        random.shuffle(self.tot_imgs_seen)

        # self.tot_imgs = [m for n in self.imgs for m in n[len(n) // 3:]]
        # self.tot_labels =[m for n in self.labels for m in n[len(n) // 3:]]
        for i in range(class_num):
            self.imgs[i] = self.imgs[i][:len(self.imgs[i]) // 3]

    def num2name(self, num):
        num_str = str(num)
        return '0' * (4 - len(num_str)) + num_str

    def getImg(self, img_path):
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.tot_imgs)

    def __getitem__(self, idx):
        return self.tot_imgs[idx], self.tot_labels[idx]

    def load_train_data(self, idx):
        return self.tot_imgs[idx], self.tot_imgs_seen[idx], self.tot_labels[idx]

if __name__ == '__main__':
    '''
    folder_path = './datasets/cifar100/train_10/'
    class_num = 10
    dataset = Cifar100Dataset(folder_path, class_num)
    print(dataset.imgs[0][0].shape)
    print(len(dataset.tot_imgs))
    print(len(dataset.tot_labels))
    print(dataset.tot_labels[:500])
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.CIFAR100(
        './datasets',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))