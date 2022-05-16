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

from pl_lib import *
from utils.eval_functions import evaluate
from utils.custom_transforms import Resize, ToTensor, Normalize

from typing import *


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
    parser.add_argument('--save_root', type=str,
                        default='results/pranet', help='root path to polyp dataset')
    args = parser.parse_args()                

    datanames = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    evaluate(datanames, args.save_root, os.path.join(args.data_root, 'TestDataset'))