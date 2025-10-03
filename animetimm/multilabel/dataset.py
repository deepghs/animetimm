import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset as _timm_load_dataset
from ditk import logging
from hfutils.operate import get_hf_client
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
from timm import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm

_DEFAULT_KEYS = [
    'rating',
    'general_tags',
    'character_tags',
]


def load_dataset(repo_id: str, split: str = 'train', transforms: Optional = None, image_key: str = 'webp',
                 categories: Optional[Sequence[int]] = None, seen_tag_keys: Optional[List[str]] = None):
    dataset = _timm_load_dataset(repo_id, split=split)
    tags_info = load_tags(repo_id, categories=categories)
    tags_to_id = tags_info.tags_to_id

    def _trans(row):
        images = []
        for image in row[image_key]:
            image = load_image(image, force_background='white', mode='RGB')
            if transforms:
                image = transforms(image)
            images.append(image)
        row['image'] = images

        all_labels = []
        for json_ in row['json']:
            labels = torch.zeros(len(tags_to_id), dtype=torch.float32)
            for tag_key in (seen_tag_keys or _DEFAULT_KEYS):
                for tag in json_[tag_key]:
                    labels[tags_to_id[tag]] = 1.0
            all_labels.append(labels)
        row['labels'] = all_labels
        return row

    dataset = dataset.with_transform(_trans)
    return dataset


def _get_normalize_from_repo_id(repo_id: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    hf_client = get_hf_client()
    if hf_client.file_exists(
            repo_id=repo_id,
            repo_type='dataset',
            filename='normalize.json'
    ):
        with open(hf_client.hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                filename='normalize.json'
        ), 'r') as f:
            d_normalize = json.load(f)
        mean, std = d_normalize['mean'], d_normalize['std']
    else:
        mean, std = None, None

    return mean, std


def load_dataloader(repo_id: str, model, split: Literal['train', 'test', 'validation'] = 'train',
                    batch_size: int = 256, num_workers: int = 128, noise_level: int = 2,
                    rotation_ratio: float = 0.25, mixup_alpha: float = 0.2,
                    cutout_max_pct: float = 0.25, cutout_patches: int = 1, random_resize_method: bool = True,
                    pre_align: bool = True, align_size: int = 512, is_main_process: bool = True,
                    image_key: str = 'webp', use_test_size_when_test: bool = True,
                    categories: Optional[Sequence[int]] = None, seen_tag_keys: Optional[List[str]] = None,
                    use_normalize: bool = False):
    from .augmentation import create_transforms

    if use_normalize:
        mean, std = _get_normalize_from_repo_id(repo_id)
    else:
        mean, std = None, None

    trans, post_trans = create_transforms(
        timm_model=model,
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
        mean=mean,
        std=std,
    )
    if is_main_process:
        logging.info(f'Transforms loaded (for {split}):\n{trans}')
        logging.info(f'Post transforms loaded (for {split}):\n{post_trans}')

    if is_main_process:
        logging.info(f'Loading dataset from {repo_id!r} (for {split}) ...')
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
        labels = []
        for example in examples:
            images.append((example["image"]))
            labels.append(example["labels"])

        pixel_values = torch.stack(images)
        labels = torch.stack(labels)
        pixel_values, labels = post_trans((pixel_values, labels))
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


def load_tags(repo_id: str, categories: Optional[Sequence[int]] = None) -> TagsInfo:
    df_tags = pd.read_parquet(hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename='tags.parquet',
    ))
    if categories:
        categories = set(categories or [])
        df_tags = df_tags[df_tags['category'].isin(categories)]
    d_min_counts = {}
    for cate in sorted(set(df_tags['category'])):
        df_cate_tags = df_tags[df_tags['category'] == cate]
        d_min_counts[cate] = int(df_cate_tags['selected_count'].min())

    weights = []
    for _, item in df_tags.iterrows():
        min_count = d_min_counts[int(item['category'])]
        weights.append(1 / (np.log(int(item['selected_count'])) / np.log(min_count)))

    df_tags['weights'] = weights
    return TagsInfo(
        df=df_tags,
        tags_to_id={tag: id for id, tag in enumerate(df_tags['name'])},
        tags=df_tags['name'].tolist(),
        weights=df_tags['weights'].to_numpy(),
    )


if __name__ == '__main__':
    rid = 'animetimm/danbooru-wdtagger-v4-w640-ws-full'
    mid = "caformer_s36.sail_in22k_ft_in1k_384"
    model = create_model(mid, pretrained=False)
    dataloader = load_dataloader(
        repo_id=rid,
        model=model,
        split='train',
    )

    ix, ox = None, None
    for input_, output in tqdm(dataloader):
        if ix is None:
            ix = input_.shape, input_.dtype
            ox = output.shape, output.dtype
            print(ix, ox)
        else:
            assert ix == (input_.shape, input_.dtype)
            assert ox == (output.shape, output.dtype)
