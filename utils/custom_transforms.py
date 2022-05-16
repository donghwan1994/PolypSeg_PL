import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from typing import *

# All implementations are referenced to `torchvision.transforms`.

class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, map):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
            map (PIL Image or Tensor): Image of segmentation map

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias),\
             F.resize(map, self.size, self.interpolation, self.max_size, self.antialias)

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class RandomResize(torch.nn.Module):
    """Resize the input image with randomly chosen size.
        Args:
            size_list (List[int]): Size list 
        
        Example:
        >>> transform = RandomResize([168, 224, (256, 280), 384])
        >>> image, target = transform(image, target)
    """
    def __init__(self, size_list, interpolation=F.InterpolationMode.BILINEAR, 
                max_size=None, antialias=None) -> None:
        super().__init__()
        self.size_list = size_list
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img, map):
        size = random.choice(self.size_list)

        return F.resize(img, size, self.interpolation, self.max_size, self.antialias), \
            F.resize(map, size, self.interpolation, self.max_size, self.antialias)


class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, img, map):
        return F.normalize(img, self.mean, self.std, self.inplace), map

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomCrop(torch.nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def forward(self, img, map):
        i, j, h, w = transforms.RandomCrop.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(map, i, j, h, w)


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, map):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(map)
        return img, map

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, map):
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(map)
        return img, map

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotate(torch.nn.Module):
    def __init__(
        self, degrees, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        super().__init__()
        self.degrees = transforms.transforms._setup_angle(degrees, name="degrees", req_sizes=(2,))

        self.center = center

        if center is not None:
           transforms.transforms._check_sequence_input(center, "center", req_sizes=(2,))

        self.resample = self.interpolation = interpolation
        self.expand = expand

        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        angle = transforms.RandomRotation.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center, fill), \
            F.rotate(map, angle, self.resample, self.expand, self.center, fill)


class RandomRotate90(torch.nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.angles = [-90, 0, 90, 180]

    def forward(self, img, map):
        if torch.rand(1) < self.p:
            angle = random.choice(self.angles)
            return F.rotate(img, angle), F.rotate(map, angle)
        return img, map

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class ToTensor:
    def __init__(self) -> None:
        pass 

    def __call__(self, img, map):
        return F.to_tensor(img), F.to_tensor(map)
