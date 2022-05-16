import os
import argparse
from imageio import save

import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from dataloader.dataset import PolypDataset
from dataloader.dataloader import PolypDataModule

from pl_lib import *
from utils.eval_functions import evaluate
from utils.custom_transforms import Resize, ToTensor, Normalize

from typing import *


def save_prediction(preds: List[Any], save_dir: str):
    for pred in preds:
        pred_map, file_name = pred
        
        pred_map = pred_map.data.cpu().numpy().squeeze()
        pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
        pred_map = ((pred_map > 0.5) * 255).astype(np.uint8)
        
        Image.fromarray(pred_map).save(os.path.join(save_dir, file_name[0]))

def test(args):
    if 'pranet' in str(args.method):
        hparams = {
            'channels': 32,
            'epoch': 2,
            'batch_size': 2,
            'lr': 1e-4,
            'decay_rate': 0.1,
            'decay_epoch': 50,
            'grad_clip_val': 0.5,
            'grad_clip_algorithm': 'value'
        }
        model = PraNet(hparams)
    elif 'sanet' in str(args.method):
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
    else:
        raise RuntimeError("The method " + str(args.method) + " is not supported.")

    trainer = Trainer(accelerator='gpu', devices=args.gpus, logger=False)
    transforms = [
        Resize((352, 352)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
    ]

    result_path = 'results'
    save_root = os.path.join(result_path, args.method)
    checkpoint_path = os.path.join('checkpoints', str(args.method), 'last.pth')
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
    parser.add_argument('--eval', type=bool, default=True, help='the decision whether to evaluation')
    args = parser.parse_args()                

    test(args)