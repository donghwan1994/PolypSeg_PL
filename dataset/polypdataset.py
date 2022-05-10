import os
import random

from PIL import Image, ImageCms, ImageStat
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from typing import *


class PolypDataset(data.Dataset):
    def __init__(
        self, 
        root: str, 
        train: bool = True,
        transforms: Optional[Callable] = None,
        dataname: str = 'Kvasir',
        color_exchange: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.root = os.path.join(root, 'TrainDataset') if train else os.path.join(root, 'TestDataset', dataname)
        self.images = [os.path.join(self.root, 'images', f) for f in os.listdir(os.path.join(self.root, 'images')) \
            if f.endswith('.jpg') or f.endswith('.png')]
        self.targets = [os.path.join(self.root, 'masks', f) for f in os.listdir(os.path.join(self.root, 'masks')) \
            if f.endswith('.png')]

        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

        assert len(self.images) == len(self.targets)
        self.filter_files()

        self.transforms = transforms
        self.color_exchange = color_exchange

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index]).convert('L')

        if self.color_exchange:
            r_indexs = [i for i in range(0, len(self.images))]
            r_index = random.choice(r_indexs)
            r_image = Image.open(self.images[r_index])
            image = self.color_exchange_fn(image, r_image)

        if self.transforms is not None:
            for t in self.transforms:
                image, target = t(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def filter_files(self):
        assert len(self.images) == len(self.targets)
        images = []
        targets = []
        for image_path, target_path in zip(self.images, self.targets):
            img = Image.open(image_path)
            gt = Image.open(target_path)
            if img.size == gt.size:
                images.append(image_path)
                targets.append(target_path)
        self.images = images
        self.targets = targets

    def color_exchange_fn(self, image1, image2):
        # RGB to LAB
        rgb_pf = ImageCms.createProfile("sRGB")
        lab_pf = ImageCms.createProfile("LAB")
        rgb2lab = ImageCms.buildTransformFromOpenProfiles(rgb_pf, lab_pf, "RGB", "LAB")
        lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_pf, rgb_pf, "LAB", "RGB")
        image1_lab = ImageCms.applyTransform(image1, rgb2lab)
        image2_lab = ImageCms.applyTransform(image2, rgb2lab)

        # Color exchange proposed SANet
        image1_lab = transforms.PILToTensor()(image1_lab)
        image2_lab = transforms.PILToTensor()(image2_lab)

        image1_lab = image1_lab.float()
        image2_lab = image2_lab.float()

        std1, mean1 = torch.std_mean(image1_lab, dim=(1, 2), keepdim=True)
        std2, mean2 = torch.std_mean(image2_lab, dim=(1, 2), keepdim=True)

        image1 = ((image1_lab - mean1) / std1 * std2 + mean2).type(torch.uint8)

        image1 = transforms.ToPILImage()(image1)

        # LAB to RGB
        image1 = ImageCms.applyTransform(image1, lab2rgb)

        return image1