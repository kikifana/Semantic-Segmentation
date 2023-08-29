# imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torch.autograd import Variable
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, Normalize, RandomHorizontalFlip,RandomResizedCrop,RandomRotation

from sklearn.metrics import confusion_matrix, classification_report
import os

from typing import List
import PIL
from PIL import Image
import warnings

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from lightning_fabric.accelerators.cuda import is_cuda_available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision.datasets import Cityscapes
from torchvision import datapoints
import torchvision.transforms.v2 as transformsv2
from torchvision.transforms.v2 import ConvertImageDtype
from torchvision.transforms.v2 import InterpolationMode
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
from torchmetrics.classification import MulticlassAccuracy
from torch.optim import Adam, SGD, LBFGS, Adadelta, Adamax, Adagrad
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.profilers import PyTorchProfiler
# import DeepLabV3 class
from deeplabv3 import  DeepLabV3

path = '/mnt/datasets/Cityscapes'

MODEL_NAME = 'deeplabv3-mobilenet-CELoss-1.0'
MODEL_TYPE = 'deeplabv3'

# classes to ignore 
_IGNORE_IDS = [-1,0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
_EVAL_IDS =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
_TRAIN_IDS =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18]

torch.set_float32_matmul_precision('high')

# class to remove unwanted ids 

class EvalToTrainIds():
    def __init__(self,
                 ignore_ids = _IGNORE_IDS,
                 eval_ids = _EVAL_IDS,
                 train_ids = _TRAIN_IDS
                 ):
        self.ignore_ids = ignore_ids
        self.eval_ids = eval_ids
        self.train_ids = train_ids

    def __call__(self, target):
        for id in self.ignore_ids:
            target = torch.where(target==id, 34, target)
        for train_id, eval_id in zip(self.train_ids, self.eval_ids):
            target = torch.where(target==eval_id, train_id, target)
        target = torch.where(target==34, 19, target)
        return target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class OneHot():
    def __init__(self, channels) -> None:
        self.channels = channels

    def __call__(self, input):
        one_hot_output = torch.zeros(self.channels, *input.shape[1:], dtype=torch.float32)
        one_hot_output.scatter_(0, torch.tensor(input.clone(), dtype=torch.int64), 1)
        return one_hot_output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

torch.manual_seed(17)

DEFAULT_TARGET_TRANSFORM = lambda num_classes: transformsv2.Compose(
    [
        EvalToTrainIds(),
        OneHot(num_classes),

    ]
)

DEFAULT_TRANSFORM = transformsv2.Compose(
    [
        
        ConvertImageDtype(torch.float32),
        #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
    ]
)

num_classes = 20
torch.manual_seed(17)

aug = transformsv2.Compose([

    transformsv2.RandomHorizontalFlip(p=0.5),
    transformsv2.CenterCrop(480),
    transformsv2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
])

class CityscapesDataset(Cityscapes):
    '''
    This class wraps image and target in datapoints.Image and datapoints.Mask
    objects respectively in order to use tranforms.v2 API for augmentation. The augmentations
    are performed fisrt followed by transform and target_transform, applied to image and target respectively
    in order to, for example, normalize image values to [0,1] and to convert target to one-hot encoding.
    '''
    def __init__(self,
                 root: str,
                 split: str = "train",
                 mode: str = "fine",
                 target_type: List[str] | str = "semantic",
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None,
                 augmentations: Callable[..., Any] | None = None
                 ) -> None:

        super().__init__(root, split, mode, target_type, transform, target_transform)
        self.augmentations = augmentations
        self.num_classes = 20

    def __getitem__(self, index: int) -> Tuple[datapoints.Image, datapoints.Mask]:

        image = Image.open(self.images[index]).convert("RGB")

        targets = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        # Wrap image and target in datapoints Image and Mask objects
        image, target = datapoints.Image(image), datapoints.Mask(target)

        if self.augmentations is not None:
            image, target =self.augmentations(image,target)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
    
train_dataset = CityscapesDataset(path, split = 'train',mode = 'fine',target_type = 'semantic', transform =DEFAULT_TRANSFORM ,target_transform =  DEFAULT_TARGET_TRANSFORM(num_classes))
val_dataset = CityscapesDataset(path, split = 'val',mode = 'fine',target_type = 'semantic', transform =DEFAULT_TRANSFORM ,target_transform =  DEFAULT_TARGET_TRANSFORM(num_classes))


train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle= True, num_workers=4,pin_memory=True,drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=24, shuffle=False, num_workers=1,pin_memory=True,drop_last=True)



model = DeepLabV3(
    model = deeplabv3_mobilenet_v3_large(num_classes=20),
    learning_rate = 1e-3,
    
    loss = nn.CrossEntropyLoss(),
)
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=10,
    verbose=False,
    mode="min",
)

model_checkpoint_path = f'saved_models/{MODEL_TYPE}/{MODEL_NAME}'
model_checkpoint_callback = ModelCheckpoint(dirpath='/mnt/logs',
                                                filename=model_checkpoint_path,
                                                save_weights_only=False,
                                                monitor='val_Mean_IoU',
                                                mode='max',
                                                verbose=True)


logger = TensorBoardLogger(save_dir=f'/mnt/logs/Tensorboard_logs',
                               name=f'{MODEL_TYPE}',
                               version=f'{MODEL_NAME}',
                               default_hp_metric=False)
progress_bar = RichProgressBar(
    theme = RichProgressBarTheme(
        description="#cc3399",
        progress_bar= "#6206e0",
        progress_bar_finished= "#6206e0",
        batch_progress="#6206e0",
        time = "#99ccff",
        processing_speed = "#ccccff",
        metrics = "#cc0066"
    )
)



trainer = pl.Trainer(

    
    callbacks=[ModelSummary(max_depth=-1),
              model_checkpoint_callback,
              progress_bar,
              early_stopping
              ]    ,
    
    profiler = "simple"   ,      
    logger = logger ,
    max_epochs=40,
    
    
    


)

trainer.fit(model, train_loader, val_loader)




