import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataloader.dataset import PolypDataset
from utils.custom_transforms import Resize, ToTensor, Normalize

from typing import *


class PolypDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 1,
        color_exchange: Optional[bool] = False, 
        train_transforms: Optional[Callable] = None, 
        val_transforms: Optional[Callable] = None, 
        test_transforms: Optional[Callable] = None,
        datanames: List[str] = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.color_exchange = color_exchange
        self.datanames = datanames

        base_transforms = [
            Resize((352, 352)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
        ]
        
        self.train_transforms = train_transforms if train_transforms is not None else base_transforms
        self.val_transforms = val_transforms if val_transforms is not None else base_transforms
        self.test_transforms = test_transforms if test_transforms is not None else base_transforms

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, 'fit'):
            self.train_dataset = PolypDataset(self.data_dir, train=True, 
                                            transforms=self.train_transforms, color_exchange=self.color_exchange)
            self.val_dataset = PolypDataset(self.data_dir, train=False, dataname='test',
                                            transforms=self.val_transforms, color_exchange=False)
        if stage in (None, 'test', 'predict'):
            self.test_datasets = {}
            for dataname in self.datanames:
                self.test_datasets[dataname] = PolypDataset(self.data_dir, train=False, dataname=dataname,
                                            transforms=self.test_transforms, color_exchange=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        loaders = {}
        for dataname, dataset in self.test_datasets.items():
            loaders[dataname] = DataLoader(dataset, 1, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return loaders

    def predict_dataloader(self):
        return self.test_dataloader()