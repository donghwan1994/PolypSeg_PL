from email import parser
import os
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
transforms.RandomCrop
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from dataset.polypdataset import PolypDataset

from pl_lib import *
from utils.custom_transforms import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,
                         default='pranet', help='polyp segmentation method, pranet|sanet|msnet')
    parser.add_argument('--epoch', type=int, 
                        default=20, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='number of workers for dataloader')
    parser.add_argument('--data_root', type=str,
                        default='/workspace/donghwan/dataset/PolypDataset/dataset', help='root path to polyp dataset')
    opt = parser.parse_args()
    
    train_transforms = None
    val_transforms = None
    if opt.method == 'pranet':
        hparams = {
            'channels': 32,
            'epoch': 20,
            'batch_size': 16,
            'lr': 1e-4,
            'decay_rate': 0.1,
            'decay_epoch': 50,
            'grad_clip_val': 0.5,
            'grad_clip_algorithm': 'value'
        }
        model = PraNet(hparams)
        train_transforms = [
            Resize((352, 352)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
        ]
    elif opt.method == 'sanet':
        hparams = {
            'channels': 64,
            'epoch': 1,
            'batch_size': 2,
            'lr': 0.4,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'nesterov': True,
            'lr_gamma': 0.5,
            'milestones': [64, 96],
            'grad_clip_val': None,
            'grad_clip_algorithm': 'norm',

        }
        model = SANet(hparams)
        train_transforms = [
            Resize((352, 352)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotate90(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
        ]
    elif opt.method == 'msnet':
        pass
    else:
        raise RuntimeError()
    
    val_transforms = [
        Resize((352, 352)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
    ]

    train_dataset = PolypDataset(opt.data_root, train=True, transforms=train_transforms, color_exchange=True)
    val_dataset = PolypDataset(opt.data_root, train=False, transforms=val_transforms, dataname='test')
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True,
                             num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)

    trainer = Trainer(
        default_root_dir=os.path.join('checkpoint', opt.method),
        gpus=1,
        max_epochs=hparams['epoch'], 
        gradient_clip_algorithm=hparams['grad_clip_algorithm'], 
        gradient_clip_val=hparams['grad_clip_val'],
        logger=TensorBoardLogger("logs/", name=opt.method),
        callbacks=[
            ModelCheckpoint(dirpath=os.path.join('checkpoint', opt.method),
                            save_weights_only=True, save_on_train_epoch_end=True)
        ]
    )

    trainer.fit(model, train_loader, val_loader)