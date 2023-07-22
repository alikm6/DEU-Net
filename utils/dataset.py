import random
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as VT
from torchvision.transforms import functional as VF


class SkinLesionDataset(Dataset):
    def __init__(
            self,
            images_file_path: str,
            masks_file_path: str,
            width: int,
            height: int,
            mode: str = 'train',
            augmentation_cfg: dict or None = None,
    ):
        self.width = width
        self.height = height
        self.mode = mode
        self.augmentation_cfg = augmentation_cfg

        images_path = SkinLesionDataset.load_txt(images_file_path)
        masks_path = SkinLesionDataset.load_txt(masks_file_path)

        assert len(images_path) == len(masks_path) and len(images_path) != 0

        self.images_path = images_path
        self.masks_path = masks_path

        self.num_images = len(images_path)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img = SkinLesionDataset.load(self.images_path[index])
        mask = SkinLesionDataset.load(self.masks_path[index])

        img = SkinLesionDataset.preprocess(img, is_mask=False, width=self.width, height=self.height)
        mask = SkinLesionDataset.preprocess(mask, is_mask=True, width=self.width, height=self.height)

        p_transform = random.random()
        if self.mode == 'train' and \
                self.augmentation_cfg and \
                self.augmentation_cfg['enable'] and \
                p_transform <= self.augmentation_cfg['prob']:

            mask = mask[None, ...]

            if random.random() < self.augmentation_cfg['hflip_prob']:
                img = VF.hflip(img)
                mask = VF.hflip(mask)

            if random.random() < self.augmentation_cfg['vflip_prob']:
                img = VF.vflip(img)
                mask = VF.vflip(mask)

            rotation_degree = random.randint(self.augmentation_cfg['rotation_range'][0],
                                             self.augmentation_cfg['rotation_range'][1])
            if rotation_degree != 0:
                img = VF.rotate(img, rotation_degree)
                mask = VF.rotate(mask, rotation_degree)

            color_jitter_transform = VT.ColorJitter(brightness=self.augmentation_cfg['brightness'],
                                                    contrast=self.augmentation_cfg['contrast'],
                                                    saturation=self.augmentation_cfg['saturation'],
                                                    hue=self.augmentation_cfg['hue'])
            img = color_jitter_transform(img)

            mask = mask.squeeze(0)

        return img, mask

    @staticmethod
    def load_txt(filename: str):
        with open(filename) as file:
            images_path = file.readlines()
            images_path = [line.rstrip() for line in images_path]

        return images_path

    @staticmethod
    def load(filename: str):
        """
        :rtype: PIL.Image
        :return:  image in PIL.Image format
        """

        ext = Path(filename).suffix

        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    @staticmethod
    def preprocess(img_pil: Image, is_mask: bool, width: Optional[int], height: Optional[int]):
        """
        :rtype: torch.Tensor
        :return:  image in torch.Tensor format
        """

        if is_mask:
            img_pil = img_pil.convert('L')

            if width is not None and height is not None:
                img_pil = img_pil.resize((width, height), Image.NEAREST)

            img_ndarray = np.array(img_pil, np.float32)
            img_ndarray = img_ndarray / 255.
            img_ndarray = img_ndarray.astype(np.int64)

            img_tensor = torch.as_tensor(img_ndarray.copy()).long().contiguous()
        else:
            img_pil = img_pil.convert('RGB')

            if width is not None and height is not None:
                img_pil = img_pil.resize((width, height), Image.BICUBIC)

            img_ndarray = np.array(img_pil, np.float32)
            # img_ndarray = np.clip(img_ndarray - np.median(img_ndarray) + 127, 0, 255)
            img_ndarray = img_ndarray / 255.
            img_ndarray = np.transpose(img_ndarray, (2, 0, 1))

            img_tensor = torch.as_tensor(img_ndarray.copy()).float().contiguous()

        return img_tensor
