import os
import argparse
import numpy as np
from datetime import datetime
from PIL import Image

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from dataloader.dataset import PolypDataset

from pl_lib import *
from utils.custom_transforms import *
from utils.eval_functions import evaluate


def train(args, hparams, model, train_transforms, exp_name):        
    val_transforms = [
        Resize((352, 352)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
    ]

    train_dataset = PolypDataset(args.data_root, train=True, transforms=train_transforms, 
                                color_exchange=hparams['color_exchange'])
    val_dataset = PolypDataset(args.data_root, train=False, transforms=val_transforms, dataname='test')
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    checkpoint_path = os.path.join('checkpoints', exp_name)
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
        amp_backend=hparams['amp_backend'], 
        amp_level=hparams['amp_level'],
        precision=hparams['precision'],
        detect_anomaly=True,
        log_every_n_steps=1,
        enable_progress_bar=args.verbose,
    )

    trainer.fit(model, train_loader, val_loader)

def save_prediction(preds: List[Any], save_dir: str):
    for pred in preds:
        pred_map, file_name = pred
        
        pred_map = pred_map.data.cpu().numpy().squeeze()
        pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
        pred_map = ((pred_map > 0.5) * 255).astype(np.uint8)
        
        Image.fromarray(pred_map).save(os.path.join(save_dir, file_name[0]))

def test(args, exp_name):
    trainer = Trainer(accelerator='gpu', devices=args.gpus, logger=False)
    transforms = [
        Resize((352, 352)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
    ]

    save_root = os.path.join('results', exp_name)
    checkpoint_path = os.path.join('checkpoints', exp_name, 'last.pth')
    datanames = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    for dataname in datanames:
        print ('[%s]' % (dataname))
        save_dir = os.path.join(save_root, dataname)
        os.makedirs(save_dir, exist_ok=True)
        dataset = PolypDataset(args.data_root, train=False, transforms=transforms, dataname=dataname)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        # trainer
        preds = trainer.predict(model, dataloaders=data_loader, 
                                ckpt_path=checkpoint_path, return_predictions=True)
        save_prediction(preds, save_dir)
        
    if args.eval:
        evaluate(datanames, save_root, os.path.join(args.data_root, 'TestDataset'))

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
    parser.add_argument("--verbose", default=False, action='store_true') 
    parser.add_argument('--eval', type=bool, default=True, help='the decision whether to evaluation')                  
    args = parser.parse_args()    

    if 'pranet' in str(args.method):
        hparams = {
            'channels': 32,
            'epoch': 20,
            'batch_size': 16,
            'lr': 1e-4,
            'decay_rate': 0.1,
            'decay_epoch': 50,
            'grad_clip_val': 0.5,
            'grad_clip_algorithm': 'value',
            'amp_backend': 'native',
            'amp_level': None,
            'precision': 32,
            'color_exchange': False
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
            'amp_backend': 'apex',
            'amp_level': '02',
            'precision': 32,
            'color_exchange': True
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
            'coeff': 0.1, 
            'epoch': 50,
            'batch_size': 16,
            'lr': 0.05,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'nesterov': True,
            'lr_gamma': 0.5,
            'grad_clip_val': None,
            'grad_clip_algorithm': 'norm',
            'amp_backend': 'native',
            'amp_level': None,
            'precision': 32,
            'color_exchange': False
        }
        model = MSNet(hparams)
        train_transforms = [
            Resize((352, 352)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotate(degrees=10),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
        ]
    elif 'prototype' in str(args.method):
        hparams = {
            'channels': 64,
            'epoch': 128,
            'batch_size': 2,
            'lr': 0.4,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'nesterov': True,
            'lr_gamma': 0.5,
            'milestones': [64, 96],
            'grad_clip_val': 0.5,
            'grad_clip_algorithm': 'value',
            'amp_backend': 'native',
            'amp_level': None,
            'precision': 32,
            'trade_off' : 0.1,
            'color_exchange': False
        }
        model = Prototype(hparams)
        train_transforms = [
            Resize((352, 352)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotate(degrees=10),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
        ]
    else:
        raise RuntimeError("The method " + str(args.method) + " is not supported.")                

    exp_name = args.method + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    train(args, hparams, model, train_transforms, exp_name)
    test(args, exp_name)
    