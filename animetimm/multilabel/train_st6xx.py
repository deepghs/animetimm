import os

from ditk import logging

from .train import train

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    set_name = str(os.environ.get('S', 'fullxx') or 'fullxx')
    max_epoch = int(os.environ.get('MEP', '100') or '100')
    mn = str(os.environ.get('M', 'hf-hub:animetimm/swinv2_base_window8_256.dbv4a-fullx')
             or 'hf-hub:animetimm/swinv2_base_window8_256.dbv4a-fullx')
    rmn = mn.replace('/', '_').replace(':', '_').replace('\\', '_')
    size = 448
    num_workers = int(os.environ.get('NW', 32) or 32)
    batch_size = int(os.environ.get('BS', 32) or 32)
    model_args = {
        "act_layer": "gelu_tanh",
        "img_size": 448,
        "window_size": 14
    }
    train(
        workdir=f'runs/st_{rmn}_{set_name}_bs{batch_size}_p512x_d0.4_mep{max_epoch}_s{size or ""}',
        dataset_repo_id=f'deepghs/danbooru-wdtagger-v4a-w640-ws-{set_name}',
        timm_model_name=mn,
        num_workers=num_workers,
        batch_size=batch_size,
        learning_rate=2e-4,
        noise_level=2,
        mixup_alpha=0.6,
        cutout_patches=1,
        cutout_max_pct=0.25,
        rotation_ratio=0.25,
        model_args=model_args,
        max_epochs=max_epoch,
        tag_categories=[1],
        seen_tag_keys=['artist_tags'],
    )
