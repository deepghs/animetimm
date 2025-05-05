from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset as _timm_load_dataset
from huggingface_hub import hf_hub_download
from timm import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_dataset(repo_id: str, tags_to_id: Dict[str, int], split: str = 'train', transforms: Optional = None):
    dataset = _timm_load_dataset(repo_id, split=split)

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
            for tag in [*json_['rating'], *json_['general_tags'], *json_['character_tags']]:
                labels[tags_to_id[tag]] = 1.0
            all_labels.append(labels)
        row['labels'] = all_labels
        return row

    dataset = dataset.with_transform(_trans)
    return dataset


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
    rid = 'animetimm/danbooru-wdtagger-v4-w640-ws-50k'
    tags_info = load_tags(rid)

    mid = "resnet50"
    model = create_model(mid, pretrained=False)
    from .augmentation import create_transforms

    trans = create_transforms(
        timm_model=model,
        is_training=True,
        # use_test_size=True,
        cutout_patches=1,
    )
    print(trans)

    dataset = load_dataset(
        repo_id=rid,
        split='train',
        tags_to_id=tags_info.tags_to_id,
        transforms=trans,
    )

    print(dataset[0])

    for i in tqdm(range(1000)):
        _ = dataset[i]


    def collate_fn(examples):
        images = []
        labels = []
        for example in examples:
            images.append((example["image"]))
            labels.append(example["labels"])

        pixel_values = torch.stack(images)
        labels = torch.stack(labels)
        return {"pixel_values": pixel_values, "labels": labels}


    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=16)

    for x in tqdm(dataloader):
        pass
