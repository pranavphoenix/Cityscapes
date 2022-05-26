# -*- coding: utf-8 -*-


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

# from google.colab import drive
# drive.mount('/content/drive')

import sys, os, time, pickle
import numpy as np
import math
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

import torch.optim as optim
!pip install torchsummary
from torchsummary import summary
!pip install einops
from math import ceil
import pywt

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
from einops import reduce

!wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=USERNAME&password=PASSWORD&submit=Login' https://www.cityscapes-dataset.com/login/
!wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
!wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1


batch_size = 64

!unzip -q /content/gtFine_trainvaltest.zip
!unzip -q /content/leftImg8bit_trainvaltest.zip

dataset = torchvision.datasets.Cityscapes('/content/', split='train', mode='fine',
                      target_type='semantic')

dataset[0][0].size

fig,ax=plt.subplots(ncols=2,figsize=(12,8))
ax[0].imshow(dataset[0][0])
ax[1].imshow(dataset[0][1],cmap='gray')

ignore_index=255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']


class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)
class_map

colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(range(n_classes), colors))

def encode_segmap(mask):
    #remove unwanted classes and recitify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def decode_segmap(temp):
    #convert gray scale to color
    temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

!pip uninstall opencv-python-headless==4.5.5.62

!pip install opencv-python-headless==4.1.2.30

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform=A.Compose(
[
    A.Resize(256, 256),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]
)

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes

class MyClass(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed=transform(image=np.array(image), mask=np.array(target))            
        return transformed['image'],transformed['mask']

dataset=MyClass('/content/', split='val', mode='fine',
                     target_type='semantic',transforms=transform)
img,seg= dataset[20]
print(img.shape,seg.shape)

fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,8))
ax[0].imshow(img.permute(1, 2, 0))
ax[1].imshow(seg,cmap='gray')

print(torch.unique(seg))
print(len(torch.unique(seg)))

res=encode_segmap(seg.clone())
print(res.shape)
print(torch.unique(res))
print(len(torch.unique(res)))

res1=decode_segmap(res.clone())

fig,ax=plt.subplots(ncols=2,figsize=(12,10))  
ax[0].imshow(res,cmap='gray')
ax[1].imshow(res1)

!pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp

!pip uninstall torchmetrics
!pip install torchmetrics==0.7.3
import torchmetrics
###################MODEL################################################################

model.to(device)
print(summary(model, (3,256,256)))    
print(torch.cuda.get_device_properties(device))

criterion= smp.losses.DiceLoss(mode='multiclass')
metrics = torchmetrics.IoU(num_classes=n_classes)

trainset = MyClass('/content/', split='train', mode='fine',
                     target_type='semantic',transforms=transform)
testset= MyClass('/content/', split='val', mode='fine',
                     target_type='semantic',transforms=transform)

batch_size = 96

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

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

# criterion = FocalLoss2d()
scaler = torch.cuda.amp.GradScaler()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
miou = []
epoch_losses = []
test_losses = []
# optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
for epoch in range(1000):  # loop over the dataset multiple times
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0

    model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        segment=encode_segmap(labels)

        with torch.cuda.amp.autocast():
            loss = criterion(outputs, segment.long())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        acc = (outputs.argmax(dim=1) == labels).float().mean()
        epoch_accuracy += acc / len(trainloader)
        epoch_loss += loss / len(trainloader)
    
        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    epoch_losses.append(epoch_loss)
    test_loss = 0
    total = 0

    mIoU = 0
    model.eval()
    with torch.no_grad():
        t1 = time.time()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # labels = labels.squeeze(1).long()
            outputs = model(images)
            segment=encode_segmap(labels)
#         outputs = net(images)

            # _, predicted = torch.max(outputs.data, 1)
            with torch.cuda.amp.autocast():
              test_loss += criterion(outputs, segment.long())
              # mIoU = metrics(outputs, labels)
              # mIoU +=  metrics(outputs,segment)
            mIoU += iou_pytorch(outputs, segment).mean()
          
    
    mIoU = mIoU/len(testloader)
    test = test_loss/len(testloader)
    test_losses.append(test)
    mIoU = mIoU.cpu().detach()
    miou.append(mIoU)

    print(f"Epoch : {epoch+1} - train loss : {epoch_loss:.4f} - test loss: {test_loss:.4f} - MIOU: {mIoU:.4f} - Train Time: {t1 - t0:.2f} -Test Time: {time.time() - t1:.2f} \n")
    if mIoU >= max(miou):
        PATH = 'model.pth'
        torch.save(model.state_dict(), PATH)
        print(1)
    
print('Finished Training')

#Evaluation

test_class = MyClass('/content/', split='val', mode='fine',
                     target_type='semantic',transforms=transform)
test_loader=torch.utils.data.DataLoader(test_class, batch_size=12, 
                      shuffle=False)


model.eval()
with torch.no_grad():
    for batch in test_loader:
        img,seg=batch
        output=model(img.cuda())
        break
print(img.shape,seg.shape,output.shape)

from torchvision import transforms
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
)

sample=4
invimg=inv_normalize(img[sample])
outputx=output.detach().cpu()[sample]
encoded_mask=encode_segmap(seg[sample].clone()) #(256, 512)
decoded_mask=decode_segmap(encoded_mask.clone())  #(256, 512)
decoded_ouput=decode_segmap(torch.argmax(outputx,0))
fig,ax=plt.subplots(ncols=3,figsize=(16,50),facecolor='white')  
ax[0].imshow(np.moveaxis(invimg.numpy(),0,2)) #(3,256, 512)
#ax[1].imshow(encoded_mask,cmap='gray') #(256, 512)
ax[1].imshow(decoded_mask) #(256, 512, 3)
ax[2].imshow(decoded_ouput) #(256, 512, 3)
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[0].set_title('Input Image')
ax[1].set_title('Ground mask')
ax[2].set_title('Predicted mask')
plt.savefig('result.png',bbox_inches='tight')
