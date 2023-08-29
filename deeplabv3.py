from torchvision.utils import draw_segmentation_masks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from typing import List
import PIL
from PIL import Image
import warnings


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

from torchvision.datasets import Cityscapes
from torchvision.models.segmentation.deeplabv3 import deeplabv3_mobilenet_v3_large
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateFinder
import torchmetrics
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification import F1Score
from torchmetrics.classification import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from lightning.pytorch.callbacks import ModelSummary

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

_EVAL_IDS_ =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33, 0] # MAP VOID CLASS TO 0 -> TOTAL BLACK
_TRAIN_IDS_ =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]

cityscapes_color_map =  {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64,128),
    8: (244, 35,232),
    9: (250,170,160),
    10: (230,150,140),
    11: ( 70, 70, 70),
    12: (102,102,156),
    13: (190,153,153),
    14: (180,165,180),
    15: (150,100,100),
    16: (150,120, 90),
    17: (153,153,153),
    18: (153,153,153),
    19: (250,170, 30),
    20: (220,220,  0),
    21: (107,142, 35),
    22: (152,251,152),
    23: (70,130,180),
    24: (220, 20, 60),
    25: (255,  0,  0),
    26: (0,  0,142),
    27: (0,  0, 70),
    28: (0, 60,100),
    29: (0, 60,100),
    30: (0,  0,110),
    31: (0, 80,100),
    32: (0,  0,230),
    33: (119, 11, 32)
}

class DeepLabV3(pl.LightningModule):
    def __init__(self,
                 learning_rate: None ,
                 model: nn.Module,
                 loss: nn.Module,
                 optimizer_config: dict = None,
                 ) -> None:

        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer_config = optimizer_config
        self.training_step_outputs = []
        #self.accuracy = torchmetrics.Accuracy(task='multiclass',
                                     #num_classes=n_classes) #
        self.train_acc = Accuracy(task='multiclass',
                                     num_classes=20,average='macro',ignore_index=19)
        self.valid_acc = Accuracy(task='multiclass',
                                     num_classes=20,average='macro',ignore_index=19)
        self.train_f1_score = F1Score(task = 'multiclass',
                                             num_classes = 20, average = 'macro',ignore_index =19)
        self.val_f1_score =F1Score(task = 'multiclass',
                                             num_classes = 20, average = 'macro',ignore_index =19)
        self.train_mean_iou = JaccardIndex(task='multiclass', 
                                         num_classes=20,
                                         average='macro',
                                         ignore_index=19)
        self.val_mean_iou = JaccardIndex(task='multiclass', 
                                         num_classes=20,
                                         average='macro',
                                         ignore_index=19)
        

    def forward(self,x):
      return self.model(x)

    def training_step(self, train_batch, batch_idx):

        input, target = train_batch

        pred = self.model(input)['out']

        loss = self.loss(pred, target)
        self.training_step_outputs.append(loss)
        
        self.train_acc(torch.argmax(pred,dim = 1), torch.argmax(target,dim=1))
        self.train_f1_score(torch.argmax(pred,dim = 1), torch.argmax(target,dim=1))
        self.train_mean_iou(torch.argmax(pred,dim = 1), torch.argmax(target,dim=1))
        self.log_dict({'train_loss': loss,'train_accuracy':self.train_acc ,'train_Mean_IoU': self.train_mean_iou, 'train-f1-score':self.train_f1_score},
                      on_step = False, on_epoch = True, prog_bar = True)
        return loss


    #def on_training_epoch_end(self, outputs):
        # log epoch metric
        #self.train_acc.reset()

    def validation_step(self, val_batch, batc_idx):
        input, target = val_batch
        pred = self.model(input)['out']
        loss = self.loss(pred, target)
        
        self.valid_acc(torch.argmax(pred,dim = 1), torch.argmax(target,dim=1))
        self.val_f1_score(torch.argmax(pred,dim = 1), torch.argmax(target,dim=1))
        self.val_mean_iou(torch.argmax(pred,dim = 1), torch.argmax(target,dim=1))
        self.log_dict({'val_loss': loss, 'val_accuracy':self.valid_acc,'val_Mean_IoU': self.val_mean_iou, 'val-f1-score':self.val_f1_score},
                      on_step = False, on_epoch = True, prog_bar = True)
        return loss
        # self.log('val_Mean_IoU', self.val_mean_iou, on_epoch=True, 
        #          on_step=False, sync_dist=True, prog_bar=True)
        # self.log('val_loss', loss, on_epoch = True)
        
        #self.log('valid_acc',self.valid_acc, on_step = True, on_epoch = True)
    #def on_validation_epoch_end(self):
        # log epoch metric
        #self.log('valid_acc_epoch', self.valid_acc.compute())
        #self.valid_acc.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 1e-3)
        return optimizer

    def predict_step(self, predict_batch: dict, batch_idx: int, dataloader_idx: int = 0):
        input = predict_batch.get('image')
        
        predictions = self.model(input)['out']
        predictions = torch.argmax(predictions, 1)
        predictions = predictions.to(dtype=torch.uint8)

        return predictions
  