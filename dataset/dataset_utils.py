import sys 
import torchvision
from torchvision import transforms
import os
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
from sklearn import preprocessing
import requests 
from io import BytesIO
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image 
from typing import Tuple,Any 
from torch.utils.data import DataLoader
from tqdm import tqdm 

def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_train_val_images_tiny(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
 
    train_dir = os.path.join(data_path, 'tiny-imagenet-200', 'train')
    test_dir = os.path.join(data_path, 'tiny-imagenet-200', 'val')
    train_dset = datasets.ImageFolder(train_dir)

    train_images = []
    train_labels = []
    for item in train_dset.imgs:
        train_images.append(item[0])
        train_labels.append(item[1])
    
    train_targets =  np.array(train_labels)
    
    test_images = []
    test_labels = []
    _, class_to_idx = find_classes(train_dir)
    imgs_path = os.path.join(test_dir, 'images')
    imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
    with open(imgs_annotations) as r:
        data_info = map(lambda s: s.split('\t'), r.readlines())
    cls_map = {line_data[0]: line_data[1] for line_data in data_info}
    for imgname in sorted(os.listdir(imgs_path)):
        if cls_map[imgname] in sorted(class_to_idx.keys()):
            path = os.path.join(imgs_path, imgname)
            test_images.append(path)
            test_labels.append(class_to_idx[cls_map[imgname]])
    
    test_targets =  np.array(test_labels)
    
    return train_images, train_targets, test_images, test_targets, class_to_idx

def get_train_val_images_imagenet_subset(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
 
    
    train_dir = os.path.join(data_path, 'imagenet_subset', 'train')
    test_dir = os.path.join(data_path, 'imagenet_subset', 'val')
    train_dset = datasets.ImageFolder(train_dir)

    train_images = []
    train_labels = []
    for item in train_dset.imgs:
        train_images.append(item[0])
        train_labels.append(item[1])
    
    train_targets =  np.array(train_labels)
    _, class_to_idx = find_classes(train_dir)
    
    val_annotations = open(os.path.join(data_path, 'imagenet_subset',   'val_100.txt'), "r")
    
    val_lines = val_annotations.readlines()
    
    test_images = []
    test_labels = []
    for item in val_lines:
        splitted_item = item.rstrip()
        splitted_item  = splitted_item.split(" ")
        image_path = "/".join([splitted_item[0].split("/")[0], splitted_item[0].split("/")[2]])
        image_path = os.path.join(data_path, 'imagenet_subset', image_path)
        test_images.append(image_path)
        test_labels.append(int(splitted_item[1]))
        
    test_targets =  np.array(test_labels)
   
    return train_images, train_targets, test_images, test_targets, class_to_idx


def get_train_val_images_imagenet_1k(data_path):
     
    train_dir = os.path.join(data_path, "imagenet", 'train')
    test_dir = os.path.join(data_path, "imagenet", 'val')
    train_dset = datasets.ImageFolder(train_dir)
    test_dset = datasets.ImageFolder(test_dir)

    train_images = []
    train_labels = []
    for item in train_dset.imgs:
        train_images.append(item[0])
        train_labels.append(item[1])
    
    train_targets =  np.array(train_labels)
    _, class_to_idx = find_classes(train_dir)
    
    
    test_images = []
    test_labels = []
    
    for item in test_dset.imgs:
        test_images.append(item[0])
        test_labels.append(item[1])

    test_targets =  np.array(test_labels)
 
    return train_images, train_targets, test_images, test_targets, class_to_idx


def get_train_val_images_domainnet(data_path, n_task):
    
    train_annotations = open(os.path.join(data_path, "DomainNet",  "cs_train_{}.txt".format(n_task)), "r")
    test_annotations = open(os.path.join(data_path, "DomainNet",  "cs_test_{}.txt".format(n_task)), "r")

    train_lines = train_annotations.readlines()
    test_lines = test_annotations.readlines() 

    train_images = []
    train_labels = []
    for item in train_lines:
        splitted_item = item.rstrip()
        splitted_item  = splitted_item.split(" ")
        image_path = "/".join([splitted_item[0].split("/")[0],splitted_item[0].split("/")[1], splitted_item[0].split("/")[2]])
        image_path = os.path.join(data_path, 'DomainNet', image_path)
        train_images.append(image_path)
        train_labels.append(int(splitted_item[1]))
    
    train_targets = np.array(train_labels)
    
    test_images = []
    test_labels = []
    for item in test_lines:
        splitted_item = item.rstrip()
        splitted_item  = splitted_item.split(" ")
        image_path = "/".join([splitted_item[0].split("/")[0], splitted_item[0].split("/")[1], splitted_item[0].split("/")[2]])
        image_path = os.path.join(data_path, 'DomainNet', image_path)
        test_images.append(image_path)
        test_labels.append(int(splitted_item[1]))
    
    test_targets = np.array(test_labels)

    return train_images, train_targets, test_images, test_targets

    
class DomainNet(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform 
        
    
    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_path, target = self.data[index], self.targets[index]
        
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            
            
        return img, target


class TinyImagenetDataset(Dataset):
    def __init__(self, data, targets, class_to_idx,  transform):
        self.data = data
        self.targets = targets
        self.transform = transform 
        self.class_to_idx = class_to_idx 
        
    
    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_path, target = self.data[index], self.targets[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')


        if self.transform is not None:
            img = self.transform(img)
            
            
        return img, target
    

        
        
def get_dataset(dataset_type, data_path, n_task):
    if  dataset_type == "cifar100":
        print("Loading Cifar 100")
        train_transform = [transforms.Pad(4), transforms.RandomResizedCrop(32), 
                           transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                           transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]  
        train_transform = transforms.Compose(train_transform)

        test_transform = [transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]  
        test_transform = transforms.Compose(test_transform)
 
  
        if dataset_type == "cifar100":
            train_set = torchvision.datasets.CIFAR100(root='./dataset/data', train=True,
                                                download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root='./dataset/data', train=False,
                                            download=True, transform=test_transform)
            n_classes = 100
 

    elif dataset_type == "tiny-imagenet":
        # images_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        print("Loading Tiny Imagenet")
  
 
        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_tiny(data_path)
        
        train_transform = transforms.Compose(
                                            [transforms.RandomCrop(64, padding=8), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                            ])
 
        
        test_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        
 
        train_set = TinyImagenetDataset(train_data, train_targets, class_to_idx, train_transform)
        test_set = TinyImagenetDataset(test_data, test_targets, class_to_idx, test_transform)
        
        n_classes = 200
        
    elif dataset_type == "imagenet-subset":
        print("Loading Imagenet Subset")
 
 
        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_imagenet_subset(data_path)
        
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        
        train_set = TinyImagenetDataset(train_data, train_targets, class_to_idx, train_transform)
        test_set = TinyImagenetDataset(test_data, test_targets, class_to_idx, test_transform)
        
        n_classes = 100
    elif dataset_type == "imagenet-1k":
        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_imagenet_1k(data_path)
        
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        
        train_set = TinyImagenetDataset(train_data, train_targets, class_to_idx, train_transform)
        test_set = TinyImagenetDataset(test_data, test_targets, class_to_idx, test_transform)
        
        n_classes = 1000
        
    elif dataset_type == "domainnet":
        
        train_data, train_targets, test_data, test_targets  = get_train_val_images_domainnet(data_path, n_task)
         
         
        train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        
        
        n_classes = len(np.unique(train_targets))
        print("Number of classes of DomainNet is {}".format(n_classes))

        train_set = DomainNet(train_data, train_targets,  train_transform)
        test_set = DomainNet(test_data, test_targets,  test_transform)
        
    
    return train_set, test_set, n_classes

 
            
        