import os
import argparse

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from dataloader.dataset import PolypDataset

from pl_lib import *
from utils.custom_transforms import Resize, RandomResize, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotate, RandomRotate90, ToTensor, Normalize


def train(args):    
    if 'pranet' in str(args.method):
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
    elif 'sanet' in str(args.method):
        hparams = {
            'channels': 64,
            'epoch': 128,
            'batch_size': 64,
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
    elif 'msnet' in str(args.method):
        hparams = {
            'channels': 64,
            'epoch': 50,
            'batch_size': 16,
            'lr': 0.05,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'nesterov': True,
            'lr_gamma': 0.5,
            'milestones': [64, 96],
            'grad_clip_val': None,
            'grad_clip_algorithm': 'norm',
        }
        model = MSNet(hparams)
        train_transforms = [
            Resize((352, 352)),
            RandomResize([224, 256, 288, 320, 352]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotate(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
        ]
    else:
        raise RuntimeError("The method " + str(args.method) + " is not supported.")
    
    val_transforms = [
        Resize((352, 352)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
    ]

    train_dataset = PolypDataset(args.data_root, train=True, transforms=train_transforms, color_exchange=True)
    val_dataset = PolypDataset(args.data_root, train=False, transforms=val_transforms, dataname='test')
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    checkpoint_path = os.path.join('checkpoints', args.method)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                            save_weights_only=True, save_on_train_epoch_end=True, save_last=True)
    checkpoint_callback.FILE_EXTENSION = ".pth"

    trainer = Trainer(
        default_root_dir=checkpoint_path,
        gpus=args.gpus,
        max_epochs=hparams['epoch'], 
        gradient_clip_algorithm=hparams['grad_clip_algorithm'], 
        gradient_clip_val=hparams['grad_clip_val'],
        logger=TensorBoardLogger("logs/", name=args.method),
        callbacks=[
            checkpoint_callback
        ],
        amp_backend="apex", 
        amp_level="O2"
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,
                         default='pranet', help='polyp segmentation method, pranet|sanet|msnet')
    parser.add_argument('--gpus', type=int,
                        default=1, help='number of gpus')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='number of workers for dataloader')
    parser.add_argument('--data_root', type=str,
                        default='/workspace/donghwan/dataset/PolypDataset/dataset', help='root path to polyp dataset')
    args = parser.parse_args()                    

    train(args)
    