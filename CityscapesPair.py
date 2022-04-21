!pip install torchsummary
!pip install einops

import sys, os, time, pickle
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import math
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from math import ceil
import pywt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from einops import reduce


class CityscapeDataset(Dataset):
    
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.label_model = label_model
        
    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        cityscape, label = self.split_image(image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label_class).long()
        return cityscape, label_class
    
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = outputs.argmax(dim=1)

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
    
def split_image(image):
    image = np.array(image)
    cityscape, label = image[:, :256, :], image[:, 256:, :]
    return cityscape, label



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

# from google.colab import drive
# drive.mount('/content/drive')

# !unzip '/content/drive/MyDrive/archive.zip'

batch_size = 32
num_classes = 19

data_dir = '/content/cityscapes_data/'
train_dir = os.path.join(data_dir, "train") 
val_dir = os.path.join(data_dir, "val")
train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)
print(len(train_fns), len(val_fns))


sample_image_fp = os.path.join(train_dir, train_fns[1])
sample_image = Image.open(sample_image_fp).convert("RGB")
plt.imshow(sample_image)
print(sample_image_fp)



sample_image = np.array(sample_image)
print(sample_image.shape)
cityscape, label = split_image(sample_image)
print(cityscape.shape, label.shape)
print(cityscape.min(), cityscape.max(), label.min(), label.max())
cityscape, label = Image.fromarray(cityscape), Image.fromarray(label)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cityscape)
axes[1].imshow(label)



color_set = set()
for train_fn in tqdm(train_fns[:10]):
    train_fp = os.path.join(train_dir, train_fn)
    image = np.array(Image.open(train_fp))
    cityscape, label = split_image(sample_image)
    label = label.reshape(-1, 3)
    local_color_set = set([tuple(c) for c in list(label)])
    color_set.update(local_color_set)
color_array = np.array(list(color_set))




label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)



cityscape, label = split_image(sample_image)
label_class = label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cityscape)
axes[1].imshow(label)
axes[2].imshow(label_class)





dataset = CityscapeDataset(train_dir, label_model)
print(len(dataset))

cityscape, label_class = dataset[0]
print(cityscape.shape, label_class.shape)


#define model


model.to(device)
print(summary(model,(3,256,256)))


trainset = CityscapeDataset(train_dir, label_model)
train_loader = DataLoader(trainset, batch_size=batch_size)

testset = CityscapeDataset(val_dir, label_model)
test_loader = DataLoader(testset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss2d()
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scaler = torch.cuda.amp.GradScaler()

epochs = 100
miou = []
epoch_losses = []
test_losses = []

#model.load_state_dict(torch.load('WaveMix256Citypair.pth'))

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in tqdm(range(epochs)):
    t0 = time.time()
    epoch_loss = 0
    test_loss = 0
    mIoU = 0
    model.train()
    for X, Y in tqdm(train_loader, total=len(train_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        with torch.cuda.amp.autocast():
            loss = criterion(Y_pred, Y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    epoch_losses.append(epoch_loss/len(train_loader))

    model.eval()
    with torch.no_grad():
        for X, Y in tqdm(test_loader, total=len(test_loader), leave=False):
          X, Y = X.to(device), Y.to(device)
          Y_pred = model(X)
          # Y_pred = torch.argmax(Y_pred, dim=1)
          with torch.cuda.amp.autocast():
            loss = criterion(Y_pred, Y)
          mIoU += iou_pytorch(Y_pred, Y).mean()
          test_loss += loss.item()
    mIoU = mIoU/len(test_loader)      
    test = test_loss/len(test_loader)
    test_losses.append(test)
    mIoU = mIoU.cpu().detach()
    miou.append(mIoU)
    print(f"Epoch : {epoch+1} - train loss : {epoch_loss:.4f} - test loss: {test_loss:.4f} - MIOU: {mIoU:.4f} - Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")
    if mIoU >= max(miou):
        PATH = 'WaveMix256Citypair.pth'
        torch.save(model.state_dict(), PATH)
        print(1)


fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].plot(epoch_losses)
axes[1].plot(test_losses)
axes[2].plot(miou)

#evaluation
model.load_state_dict(torch.load('WaveMix256Citypair.pth'))

X, Y = next(iter(test_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
miou = iou_pytorch(Y_pred, Y)
Y_pred = torch.argmax(Y_pred, dim=1)

miou = miou.tolist()
inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])


fig, axes = plt.subplots(10, 3, figsize=(3*5, 10*5))


def Nmaxelements(list1, N):
    top = []
    list2 = list1.copy()

    for i in range(N):
      indices = [i for i, x in enumerate(list1) if x == max(list1)]
      top.append(indices)
      # list2.remove(max(list2))
      # print(len(list1))
      # print(len(list2))

    return top


x  = Nmaxelements(miou, 10)
# print(x)
print(len(miou))
j = 0
for i in [i for i, x in enumerate(miou) if x == max(miou)]:
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    
    axes[j, 0].imshow(landscape)
    axes[j, 0].set_title("Landscape")
    axes[j, 1].imshow(label_class)
    axes[j, 1].set_title("Label Class")
    axes[j, 2].imshow(label_class_predicted)
    axes[j, 2].set_title("Label Class - Predicted")
    j+=1
