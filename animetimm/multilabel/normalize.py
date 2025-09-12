from typing import Literal, Optional, Sequence, List

import timm
import torch
from ditk import logging
from timm.data import MaybeToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose

from .augmentation import create_transforms
from .dataset import load_dataset


def create_normalize_transform(
        model_name: str = 'caformer_s36.sail_in22k_ft_in1k_384',
        is_training: bool = False, use_test_size: bool = False,
        noise_level: int = 2, rotation_ratio: float = 0.25, mixup_alpha: float = 0.2,
        cutout_max_pct: float = 0.25, cutout_patches: int = 1, random_resize_method: bool = True,
        pre_align: bool = True, align_size: int = 512
):
    timm_model = timm.create_model(model_name, pretrained=False)
    transform, _ = create_transforms(
        timm_model=timm_model,
        is_training=is_training,
        use_test_size=use_test_size,
        noise_level=noise_level,
        rotation_ratio=rotation_ratio,
        mixup_alpha=mixup_alpha,
        cutout_max_pct=cutout_max_pct,
        cutout_patches=cutout_patches,
        random_resize_method=random_resize_method,
        pre_align=pre_align,
        align_size=align_size
    )

    trans_items = []
    for item in transform.transforms:
        if isinstance(item, Normalize):
            trans_items.append(MaybeToTensor())
        else:
            trans_items.append(item)
    transform, origin_transform = Compose(trans_items), transform
    return transform


def load_dataloader(repo_id: str, model_name: str = 'caformer_s36.sail_in22k_ft_in1k_384',
                    split: Literal['train', 'test', 'validation'] = 'train',
                    batch_size: int = 256, num_workers: int = 128, noise_level: int = 2,
                    rotation_ratio: float = 0.25, mixup_alpha: float = 0.2,
                    cutout_max_pct: float = 0.25, cutout_patches: int = 1, random_resize_method: bool = True,
                    pre_align: bool = True, align_size: int = 512, is_main_process: bool = True,
                    image_key: str = 'webp', use_test_size_when_test: bool = True,
                    categories: Optional[Sequence[int]] = None, seen_tag_keys: Optional[List[str]] = None):
    trans = create_normalize_transform(
        model_name=model_name,
        is_training=split == 'train',
        use_test_size=use_test_size_when_test and (split == 'test'),
        noise_level=noise_level if split == 'train' else 0,
        rotation_ratio=rotation_ratio,
        mixup_alpha=mixup_alpha,
        cutout_max_pct=cutout_max_pct,
        cutout_patches=cutout_patches,
        random_resize_method=random_resize_method,
        pre_align=pre_align,
        align_size=align_size,
    )
    if is_main_process:
        logging.info(f'Transforms (normalize) loaded (for {split}):\n{trans}')

    if is_main_process:
        logging.info(f'Loading dataset (normalize) from {repo_id!r} (for {split}) ...')
    dataset = load_dataset(
        repo_id=repo_id,
        split=split,
        transforms=trans,
        categories=categories,
        seen_tag_keys=seen_tag_keys,
        image_key=image_key,
    )

    def collate_fn(examples):
        images = []
        for example in examples:
            images.append((example["image"]))

        pixel_values = torch.stack(images)
        return pixel_values

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=split == 'train',
        drop_last=split == 'train',
    )
    return dataloader


if __name__ == '__main__':
    dataloader = load_dataloader(repo_id='animetimm/danbooru-wdtagger-v4-w640-ws-full')
    for x in enumerate(dataloader):
        print(x.shape)
        quit()
