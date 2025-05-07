from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset as _timm_load_dataset
from huggingface_hub import hf_hub_download
from timm import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_dataset(repo_id: str, split: str = 'train',
                 transforms: Optional = None):
    dataset = _timm_load_dataset(repo_id, split=split)
    tags_info = load_tags(repo_id)
    tags_to_id = tags_info.tags_to_id

    def _trans(row):
        images = []
        for image in row['webp']:
            if transforms:
                image = transforms(image)
            images.append(image)
        row['image'] = images

        all_labels = []
        for json_ in row['json']:
            labels = torch.zeros(len(tags_to_id), dtype=torch.float32)
            for tag in chain(json_['rating'], json_['general_tags'], json_['character_tags']):
                labels[tags_to_id[tag]] = 1.0
            all_labels.append(labels)
        row['labels'] = all_labels
        return row

    dataset = dataset.with_transform(_trans)
    return dataset


def load_dataloader(repo_id: str, model, split: Literal['train', 'test', 'validation'] = 'train',
                    batch_size: int = 256, num_workers: int = 128):
    from .augmentation import create_transforms
    trans, post_trans = create_transforms(
        timm_model=model,
        is_training=split == 'train',
        use_test_size=split == 'test',
        cutout_patches=1,
    )
    dataset = load_dataset(
        repo_id=repo_id,
        split=split,
        transforms=trans,
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


def load_tags(repo_id: str) -> TagsInfo:
    df_tags = pd.read_parquet(hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename='tags.parquet',
    ))
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
