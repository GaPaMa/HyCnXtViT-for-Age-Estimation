import os
import cv2
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DATASET(Dataset):
    def __init__(self, path, dataset, subset='Test', transform=None):
        super().__init__()
        df = pd.read_pickle(os.path.join(path, dataset + '.pkl'))

        index_list = df[df['Fold'] == subset].index.tolist()
        df = df.iloc[index_list]

        self.images = [os.path.join(path, dataset, image) for image in df['Image'].tolist()]
        self.labels = df['Age'].tolist()
        self.transform = transform

    def __getitem__(self, item):
        #image = Image.open(self.images[item])
        image = cv2.imdecode(np.fromfile(self.images[item], dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        label = float(self.labels[item])
        return image, label

    def __len__(self):
        return len(self.labels)
    
class IMAGENET(Dataset):
    def __init__(self, path, subset='Train', transform=None):
        super().__init__()
        df = pd.read_pickle(os.path.join(path, 'ImageNet.pkl'))

        index_list = df[df['Fold'] == subset].index.tolist()
        df = df.iloc[index_list]

        self.images = [os.path.join(path, 'ImageNet', image) for image in df['Image'].tolist()]
        self.labels = df['Label'].tolist()
        self.num_classes = len(set(self.labels))
        self.transform = transform

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = cv2.imdecode(np.fromfile(self.images[item], dtype=np.float32), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.labels[item])
        return image, label

    def __len__(self):
        return len(self.labels)


def load_transforms():
    train_transform = transforms.Compose([transforms.ToPILImage('RGB'),
                                          transforms.Resize((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomGrayscale(p=0.5),
                                          #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                          #transforms.RandomRotation(degrees=10),
                                          #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                        #   RandomSquareErasing(),
                                          transforms.ToTensor(), 
                                        #   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) #config_ReduceLROnPlateau
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    eval_transform = transforms.Compose([transforms.ToPILImage('RGB'),
                                          transforms.Resize((224, 224)), 
                                          transforms.ToTensor(), 
                                        #   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) #config_ReduceLROnPlateau 
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return train_transform, eval_transform

def load_transforms_imagenet():
    train_transform = transforms.Compose([transforms.ToPILImage('RGB'),transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(), transforms.RandomGrayscale(p=0.5), 
                                        #   transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                          transforms.RandomRotation(degrees=10),
                                          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                          transforms.ToTensor(),
                                          RandomSquareErasing(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    eval_transform = transforms.Compose([transforms.ToPILImage('RGB'), transforms.Resize((256, 256)), 
                                         transforms.CenterCrop(224), transforms.ToTensor(), 
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    return train_transform, eval_transform

class RandomSquareErasing(transforms.RandomErasing):
    def __init__(self, square_prob=0.5, min_square_size=16, max_square_size=32, p=0.5):
        super().__init__(p=p)
        self.square_prob = square_prob
        self.min_square_size = min_square_size
        self.max_square_size = max_square_size

    def erase(self, img, i, j, h, w):
        if random.uniform(0, 1) < self.square_prob:
            size = random.randint(self.min_square_size, self.max_square_size)
            if h > size and w > size:
                i = random.randint(0, h - size)
                j = random.randint(0, w - size)
                return super().erase(img, i, j, size, size)
        return super().erase(img, i, j, h, w)

"""
class CutOut(object):
    def __init__(self, height=[8, 32], width=[8, 32], p=0.5):

        self.height = height
        self.width = width
        self.p = p

    def __call__(self):

        return img
"""

"""
path = "/lscratch/gmaroun/Datasets"
dataset = "MORPH2"
subset = "Test"
transform, _ = load_transforms()
# print(transform)
train = DATASET(path, dataset, subset, transform)
image, label = train[0]
print(image)
"""