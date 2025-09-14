from typing import Optional, List

import timm
import torchvision.transforms as transforms
from imgutils.preprocess.torchvision import PadToSize
from timm.data import create_transform as _timm_create_transform
from timm.data import resolve_data_config
from torchvision.transforms import InterpolationMode

from ..augmentation import RandomResizeMethod, MixupTransform, NothingChange, Cutout, sample_beta_distribution


def create_transforms(timm_model, is_training: bool = False, use_test_size: bool = False,
                      noise_level: int = 2, rotation_ratio: float = 0.25, mixup_alpha: float = 0.2,
                      cutout_max_pct: float = 0.25, cutout_patches: int = 1, random_resize_method: bool = True,
                      pre_align: bool = True, align_size: int = 512,
                      mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
    config = resolve_data_config({}, model=timm_model, use_test_size=use_test_size)
    image_size = config['input_size'][-2]
    transform_list = []
    if mean is not None:
        config['mean'] = mean
    if std is not None:
        config['std'] = std

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
    alpha = 0.6
    print(sample_beta_distribution(16, alpha, alpha))
    quit()
    model_name = "caformer_s36.sail_in22k_ft_in1k_384"  # 可以替换为任何timm支持的模型
    model = timm.create_model(model_name, pretrained=True)

    print('Train:')
    x, p = create_transforms(
        model,
        is_training=True,
        noise_level=2,
        mixup_alpha=0.6,
        cutout_patches=0,
        cutout_max_pct=0.0,
        rotation_ratio=0.0,
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
