from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Callable, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_dataset as _timm_load_dataset
from ditk import logging
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
from torch.utils.data import DataLoader


def load_dataset(repo_id: str, tag_key: str, tag_filters: Optional[dict] = None,
                 split: str = 'train', image_key: str = 'webp', transforms: Optional = None,
                 row_level_preprocess: Optional[Callable[[Image.Image, dict], Tuple[Image.Image, dict]]] = None):
    dataset = _timm_load_dataset(repo_id, split=split)
    tags_info = load_tags(repo_id, filters=tag_filters)
    tags_to_id = tags_info.tags_to_id
    row_level_preprocess = row_level_preprocess or (lambda x, y: (x, y))

    def _trans(row):
        if row_level_preprocess is not None:
            for i, (image, json_) in enumerate(zip(row[image_key], row['json'])):
                image, json_ = row_level_preprocess(image, json_)
                row[image_key][i], row['json'][i] = image, json_

        images = []
        for image in row[image_key]:
            image = load_image(image, force_background='white', mode='RGB')
            if transforms:
                image = transforms(image)
            images.append(image)
        row['image'] = images

        all_labels = []
        for json_ in row['json']:
            all_labels.append(tags_to_id[json_[tag_key]])
        row['labels'] = all_labels
        return row

    dataset = dataset.with_transform(_trans)
    return dataset


def load_dataloader(repo_id: str, model, tag_key: str, split: Literal['train', 'test', 'validation'] = 'train',
                    batch_size: int = 256, num_workers: int = 128, noise_level: int = 2,
                    rotation_ratio: float = 0.25, cutout_max_pct: float = 0.25, cutout_patches: int = 1,
                    random_resize_method: bool = True, pre_align: bool = True, align_size: int = 512,
                    is_main_process: bool = True, image_key: str = 'webp', tag_filters: Optional[dict] = None,
                    use_test_size_when_test: bool = True,
                    row_level_augmentation: Optional[Callable[[Image.Image, dict], Tuple[Image.Image, dict]]] = None,
                    row_level_preprocess: Optional[Callable[[Image.Image, dict], Tuple[Image.Image, dict]]] = None):
    from .augmentation import create_transforms
    trans = create_transforms(
        timm_model=model,
        is_training=split == 'train',
        use_test_size=use_test_size_when_test and (split == 'test'),
        noise_level=noise_level if split == 'train' else 0,
        rotation_ratio=rotation_ratio,
        cutout_max_pct=cutout_max_pct,
        cutout_patches=cutout_patches,
        random_resize_method=random_resize_method,
        pre_align=pre_align,
        align_size=align_size,
    )
    if is_main_process:
        logging.info(f'Transforms loaded (for {split}):\n{trans}')

    if is_main_process:
        logging.info(f'Loading dataset from {repo_id!r} (for {split}) ...')
    row_level_augmentation = row_level_augmentation or (lambda x, y: (x, y))
    row_level_preprocess = row_level_preprocess or (lambda x, y: (x, y))
    if split == 'train':
        def _preprocess(x, y):
            x, y = row_level_preprocess(x, y)
            x, y = row_level_augmentation(x, y)
            return x, y
    else:
        _preprocess = row_level_preprocess
    dataset = load_dataset(
        repo_id=repo_id,
        split=split,
        transforms=trans,
        image_key=image_key,
        tag_key=tag_key,
        tag_filters=tag_filters,
        row_level_preprocess=_preprocess,
    )

    def collate_fn(examples):
        images = []
        labels = []
        for example in examples:
            images.append((example["image"]))
            labels.append(example["labels"])

        pixel_values = torch.stack(images)
        labels = torch.as_tensor(labels)
        return pixel_values, labels

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=split == 'train',
        drop_last=split == 'train',
    )
    return dataloader


@dataclass
class TagsInfo:
    df: pd.DataFrame
    tags_to_id: Dict[str, int]
    tags: List[str]
    weights: np.ndarray


def load_tags(repo_id: str, filters: Optional[dict] = None, cof: float = 1.0) -> TagsInfo:
    df_tags = pd.read_parquet(hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename='tags.parquet',
    ))
    for column, value in (filters or {}).items():
        df_tags = df_tags[df_tags[column] == value]
    d_min_counts = {}
    if 'category' in df_tags.columns:
        has_category = True
        for cate in sorted(set(df_tags['category'])):
            df_cate_tags = df_tags[df_tags['category'] == cate]
            d_min_counts[cate] = int(df_cate_tags['selected_count'].min())
    else:
        has_category = False
        d_min_counts[None] = int(df_tags['selected_count'].min())

    weights = []
    for _, item in df_tags.iterrows():
        min_count = d_min_counts[int(item['category']) if has_category else None]
        weights.append((1 / (np.log(int(item['selected_count'])) / np.log(min_count))) ** cof)

    df_tags['weights'] = weights
    return TagsInfo(
        df=df_tags,
        tags_to_id={tag: id for id, tag in enumerate(df_tags['name'])},
        tags=df_tags['name'].tolist(),
        weights=df_tags['weights'].to_numpy(),
    )


if __name__ == '__main__':
    repo_id = 'deepghs/danbooru-wdtagger-v4a-w640-ws-fullxx-cls'
    f = load_tags(repo_id=repo_id, filters={'category': 1}, cof=0.0)
    print(f.df)

    ds = load_dataset(
        repo_id=repo_id,
        tag_key='artist_tag',
        tag_filters={'category': 1},
    )
    print(ds[0])
