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
from torch.autograd import Variable
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, Normalize, RandomHorizontalFlip,RandomResizedCrop,RandomRotation

import os

from typing import List
import PIL
from PIL import Image
import warnings

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

from torchvision.datasets import Cityscapes
from torchvision import datapoints
import torchvision.transforms.v2 as transformsv2
from torchvision.transforms.v2 import ConvertImageDtype
from torchvision.models.segmentation.deeplabv3 import deeplabv3_mobilenet_v3_large
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateFinder
import torchmetrics

from lightning.pytorch.callbacks import ModelSummary
from deeplabv3 import  DeepLabV3
from lightning.pytorch.callbacks import BaseFinetuning, Callback, BasePredictionWriter
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

MODEL_NAME = 'deeplabv3-mobilenet-CELoss-1.0'
MODEL_TYPE = 'deeplabv3'

path = '/mnt/datasets/Cityscapes'
grayscale_path = f'/mnt/logs/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/grayscale'
rgb_path = f'/mnt/logs/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/rgb'

logs_dir = '/mnt/logs'
checkpoint_dir = f'{logs_dir}/saved_models/{MODEL_TYPE}/{MODEL_NAME}.ckpt'

os.makedirs(grayscale_path, exist_ok=True)
os.makedirs(rgb_path, exist_ok=True)

torch.set_float32_matmul_precision('high')

progress_bar = RichProgressBar(
    theme = RichProgressBarTheme(
        description="#cc3399",
        progress_bar= "#6206e0",
        progress_bar_finished= "#6206e0",
        batch_progress="#6206e0",
        time = "#99ccff",
        processing_speed = "#ccccff",
        metrics = "#cc0066"
    ))

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

_EVAL_IDS_ =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33, 0] # MAP VOID CLASS TO 0 -> TOTAL BLACK
_TRAIN_IDS_ =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]

class CityscapesPredictionWriter(BasePredictionWriter):
    def __init__(self,
                 logs_dir: str,
                 model_arch: str,
                 model_name: str,
                 write_interval  = "batch"
                 ) -> None:
        super().__init__(write_interval)
        self.logs_dir = logs_dir
        self.grayscale_pred_save_path = os.path.join(logs_dir, 'predictions', model_arch, 
                                                     model_name, 'test', 'grayscale')
        self.rgb_pred_save_path = os.path.join(logs_dir, 'predictions', model_arch, 
                                               model_name, 'test', 'rgb')

    def write_on_batch_end(self, 
                           trainer, 
                           pl_module, 
                           prediction,
                           batch_indices, 
                           batch, 
                           batch_idx, 
                           dataloader_idx,
                           ):
        
        input = batch.get('image')
        filenames = list(batch.get('filename'))
        
        # Map back to Eval ids
        for train_id, eval_id in zip(reversed(_TRAIN_IDS_), reversed(_EVAL_IDS_)):        
            prediction = torch.where(prediction==train_id, eval_id, prediction)

        for idx, filename in enumerate(filenames):
            pred = prediction[idx].to('cpu')
            img = input[idx].to('cpu')
            img = transformsv2.functional.convert_image_dtype(img, dtype=torch.uint8)
            
            # save grayscale predictions
            grayscale_img = transformsv2.functional.to_pil_image(pred)
            grayscale_img.save(f'{self.grayscale_pred_save_path}/{filename}')
            
            # Draw segmentation mask on top of original image
            boolean_masks = pred == torch.arange(34)[:, None, None]
            overlayed_mask = draw_segmentation_masks(img, 
                                                     boolean_masks, 
                                                     alpha=0.4, 
                                                     colors=list(cityscapes_color_map.values()))
            
            overlayed_mask_img = transformsv2.functional.to_pil_image(overlayed_mask)
            overlayed_mask_img.save(f'{self.rgb_pred_save_path}/{filename}')

class CityscapesTesting(Cityscapes):
    def __init__(self,
                 root: str,
                 transform: Callable[..., Any] | None = None,
                 
                 ) -> None:

        super().__init__(root=root,
                         transform=transform,
                         split='test',
                         mode='fine',
                         target_type='semantic',
                         target_transform=None,
                         transforms=None,
                         )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, filename): The input image to be inserted to the model for prediction and the
            filename of that image.
        """
        image_path = self.images[index]
        image_filename = image_path.split('/')[-1]

        image = Image.open(image_path).convert("RGB")
        image = datapoints.Image(image)

        if self.transform is not None:
            image = self.transform(image)
   

        return {
            'image': image,
            'filename': image_filename
        }


DEFAULT_TRANSFORM = transformsv2.Compose(
    [
        ConvertImageDtype(torch.float32),
        #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
    ]
)

aug = transformsv2.Compose([

    transformsv2.RandomHorizontalFlip(p=0.5),
    transformsv2.CenterCrop(480),
    transformsv2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
])

test_class = CityscapesTesting(path,transform =DEFAULT_TRANSFORM)


test_loader=DataLoader(test_class, batch_size=8,
                      shuffle=False, num_workers=6,pin_memory=True,drop_last=True)

trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=False,
        callbacks=[CityscapesPredictionWriter(logs_dir, MODEL_TYPE, MODEL_NAME),
                   progress_bar
                   
        ]
    )



model = DeepLabV3.load_from_checkpoint(checkpoint_dir, model = deeplabv3_mobilenet_v3_large(num_classes=20),
    learning_rate = 1e-3,
    loss = nn.CrossEntropyLoss(),
)

trainer.predict(model,test_loader)

