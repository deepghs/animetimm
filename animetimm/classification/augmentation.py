import torchvision.transforms as transforms
from imgutils.preprocess.torchvision import PadToSize
from timm.data import create_transform as _timm_create_transform
from timm.data import resolve_data_config
from torchvision.transforms import InterpolationMode

from ..augmentation import RandomResizeMethod, Cutout, prob_greyscale


def create_transforms(timm_model, is_training: bool = False, use_test_size: bool = False,
                      noise_level: int = 2, rotation_ratio: float = 0.25,
                      cutout_max_pct: float = 0.25, cutout_patches: int = 1, random_resize_method: bool = True,
                      grayscale_prob: float = 0.0, pre_align: bool = True, align_size: int = 512):
    config = resolve_data_config({}, model=timm_model, use_test_size=use_test_size)
    image_size = config['input_size'][-2]
    transform_list = []

    if is_training:
        if noise_level >= 1:
            if grayscale_prob > 0:
                transform_list.append(prob_greyscale(grayscale_prob))

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
    else:
        trans = _timm_create_transform(**config, is_training=is_training)

    if pre_align:
        trans = transforms.Compose([
            PadToSize(size=align_size),
            *trans.transforms,
        ])

    return trans
