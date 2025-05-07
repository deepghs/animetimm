import random

import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from imgutils.preprocess.torchvision import PadToSize
from timm.data import create_transform as _timm_create_transform
from timm.data import resolve_data_config
from torchvision.transforms import functional as F, InterpolationMode


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = torch.tensor(np.random.gamma(concentration_1, 1, size))
    gamma_2_sample = torch.tensor(np.random.gamma(concentration_0, 1, size))
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


class RandomResizeMethod(transforms.Resize):
    def __init__(self, size):
        super().__init__(size)
        self.methods = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC]

    def forward(self, img):
        method = random.choice(self.methods)
        return F.resize(img, self.size, method)


class Cutout:
    def __init__(self, max_pct: float = 0.25, replace: float = 0.5, patches: int = 1):
        self.max_pct = max_pct
        self.replace = replace
        self.patches = patches

    def __call__(self, img):
        img_h, img_w = img.shape[-2:]
        img_area = img_h * img_w
        pad_area = img_area * self.max_pct
        pad_size = int(np.sqrt(pad_area) / 2)
        mask = torch.ones_like(img)

        for _ in range(self.patches):
            center_h = random.randint(0, img_h)
            center_w = random.randint(0, img_w)
            lower_pad = max(0, center_h - pad_size)
            upper_pad = max(0, img_h - center_h - pad_size)
            left_pad = max(0, center_w - pad_size)
            right_pad = max(0, img_w - center_w - pad_size)
            mask[:, lower_pad:img_h - upper_pad, left_pad:img_w - right_pad] = 0
            img = img * mask + self.replace * (1 - mask)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_pct={self.max_pct}, replace={self.replace}, patches={self.patches})"


class MixupTransform:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        lam = sample_beta_distribution(batch_size, self.alpha, self.alpha)
        lam = lam.to(images.device)
        indices = torch.randperm(batch_size).to(images.device)
        lam_reshaped = lam.view(-1, 1, 1, 1)
        mixed_images = lam_reshaped * images + (1 - lam_reshaped) * images[indices]
        lam_reshaped = lam.view(-1, 1)
        mixed_labels = lam_reshaped * labels + (1 - lam_reshaped) * labels[indices]
        return mixed_images, mixed_labels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class NothingChange:
    def __call__(self, x):
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def create_transforms(timm_model, is_training: bool = False, use_test_size: bool = False,
                      noise_level: int = 2, rotation_ratio: float = 0.25, mixup_alpha: float = 0.2,
                      cutout_max_pct: float = 0.25, cutout_patches: int = 1, random_resize_method: bool = True,
                      pre_align: bool = True, align_size: int = 512):
    config = resolve_data_config({}, model=timm_model, use_test_size=use_test_size)
    image_size = config['input_size'][-2]
    transform_list = []

    if is_training:
        if noise_level >= 1:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(image_size, scale=(0.87, 0.998), ratio=(1.0, 1.0)),
            ])

            if rotation_ratio > 0:
                transform_list.append(transforms.RandomRotation(
                    degrees=rotation_ratio * 180,
                    interpolation=InterpolationMode.BILINEAR,
                ))

        if random_resize_method:
            transform_list.append(RandomResizeMethod(image_size))
        else:
            transform_list.append(transforms.Resize(image_size))

        transform_list.append(transforms.ToTensor())
        if noise_level >= 2 and cutout_max_pct > 0 and cutout_patches > 0:
            transform_list.append(Cutout(max_pct=cutout_max_pct, patches=cutout_patches))
        transform_list.append(transforms.Normalize(mean=config['mean'], std=config['std']))

        trans = transforms.Compose(transform_list)
        post_trans = (MixupTransform(mixup_alpha) if mixup_alpha > 0 and noise_level >= 2 else NothingChange())
    else:
        trans, post_trans = _timm_create_transform(**config, is_training=is_training), NothingChange()

    if pre_align:
        trans = transforms.Compose([
            PadToSize(size=align_size),
            *trans.transforms,
        ])

    return trans, post_trans


if __name__ == '__main__':
    model_name = "caformer_s36.sail_in22k_ft_in1k_384"  # 可以替换为任何timm支持的模型
    model = timm.create_model(model_name, pretrained=True)

    print('Train:')
    x, p = create_transforms(
        model,
        is_training=True,
        noise_level=1,
        rotation_ratio=0.0,
        mixup_alpha=0.6,
    )
    print(x)
    print(p)
    print()

    print('Eval:')
    x, p = create_transforms(
        model,
        is_training=False,
    )
    print(x)
    print(p)
    print()

    # print(list(x))
