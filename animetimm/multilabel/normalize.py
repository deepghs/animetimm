from typing import Literal, Optional, Sequence, List

import numpy as np
import timm
import torch
from accelerate import Accelerator
from ditk import logging
from timm.data import MaybeToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
from tqdm import tqdm

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
                    batch_size: int = 256, num_workers: int = 16, noise_level: int = 2,
                    rotation_ratio: float = 0.25, mixup_alpha: float = 0.2,
                    cutout_max_pct: float = 0.25, cutout_patches: int = 1, random_resize_method: bool = True,
                    pre_align: bool = True, align_size: int = 512, is_main_process: bool = True,
                    image_key: str = 'webp', use_test_size_when_test: bool = True,
                    categories: Optional[Sequence[int]] = None, seen_tag_keys: Optional[List[str]] = None):
    trans = create_normalize_transform(
        model_name=model_name,
        is_training=False,
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
        shuffle=True,
        drop_last=False,
    )
    return dataloader


def compute_total_mean_std(means, stds, batches):
    means = np.array(means)
    stds = np.array(stds)
    batches = np.array(batches)

    total_samples = np.sum(batches)
    total_mean = np.sum(means * batches) / total_samples
    batch_variances = stds ** 2
    weighted_variance_sum = np.sum(batches * (batch_variances + means ** 2))
    total_variance = weighted_variance_sum / total_samples - total_mean ** 2
    total_std = np.sqrt(total_variance)

    return total_mean, total_std


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    dataloader = load_dataloader(
        # repo_id='animetimm/danbooru-wdtagger-v4-w640-ws-30k',
        repo_id='animetimm/danbooru-wdtagger-v4-w640-ws-full',
        is_main_process=accelerator.is_main_process
    )
    dataloader = accelerator.prepare(dataloader)
    logging.info(f'Worker #{accelerator.process_index} ready for running.')
    accelerator.wait_for_everyone()

    means, stds, batches = [], [], []
    for i, inputs in enumerate(tqdm(
            dataloader,
            # disable=not accelerator.is_local_main_process,
            disable=False,
            desc=f'Train on Rank #{accelerator.process_index}'
    )):
        means.append(inputs.mean(dim=(0, 2, 3)))
        stds.append(inputs.std(dim=(0, 2, 3)))
        batches.append(inputs.shape[0])
        if i % 20 == 0:
            accelerator.wait_for_everyone()

    means = torch.stack(means)
    stds = torch.stack(stds)
    batches = torch.tensor(batches).to(accelerator.device)
    logging.info(f'Worker #{accelerator.process_index} ready for gathering.')
    accelerator.wait_for_everyone()

    means = accelerator.gather(means)
    stds = accelerator.gather(stds)
    batches = accelerator.gather(batches)

    if accelerator.is_main_process:
        logging.info(f'Gather completed, means shape: {means.shape!r}, '
                     f'stds shape: {stds.shape!r}, batches shape: {batches.shape!r}')
        print(means.shape, stds.shape, batches.shape)

        means = means.detach().cpu().numpy()
        stds = stds.detach().cpu().numpy()
        batches = batches.detach().cpu().numpy()

        final_means, final_stds = [], []
        for i in range(3):
            logging.info(f'Calculating channel #{i} ...')
            mean_v_c, std_v_c = compute_total_mean_std(means[:, i], stds[:, i], batches)
            final_means.append(mean_v_c)
            final_stds.append(std_v_c)

        final_means, final_stds = np.array(final_means), np.array(final_stds)
        total_count = batches.sum()
        logging.info(f'Final means: {final_means!r}')
        logging.info(f'Final stds: {final_stds!r}')
        logging.info(f'Total count: {total_count!r}')
