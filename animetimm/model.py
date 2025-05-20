import copy
import json
import logging
import os.path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple

import pandas as pd
import torch
from PIL import Image
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_pack, archive_unpack
from hfutils.utils import hf_normpath
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
from imgutils.resource import random_bg_image
from safetensors import safe_open
from safetensors.torch import save_model, load_model
from timm import create_model as _timm_create_model
from timm.models import parse_model_name, split_model_name_tag, load_model_config_from_hf
from torch import nn


@dataclass
class Model:
    module: nn.Module
    model_name: str
    tags: List[str]
    model_args: dict
    pretrained_cfg: dict

    def __post_init__(self):
        model_source, model_name = parse_model_name(self.model_name)
        if model_source == 'hf-hub':
            pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_id=model_name)
            self.model_args = {**model_args, **self.model_args}
            self.pretrained_cfg = {**pretrained_cfg, **self.pretrained_cfg}

        if 'img_size' in self.model_args:
            img_size = self.model_args['img_size']
            self.pretrained_cfg['input_size'] = [3, img_size, img_size]
            self.pretrained_cfg['test_input_size'] = [3, img_size, img_size]

        self.module.pretrained_cfg.update(self.pretrained_cfg)

    @property
    def src_repo_id(self) -> str:
        model_source, model_name = parse_model_name(self.model_name)
        if model_source == 'hf-hub':
            return model_name
        else:
            return f'timm/{model_name}'

    @property
    def architecture(self) -> str:
        model_source, model_name = parse_model_name(self.model_name)
        if model_source == 'hf-hub':
            with open(hf_hub_download(repo_id=self.src_repo_id, repo_type='model', filename='config.json'), 'r') as f:
                meta = json.load(f)
                return meta['architecture']
        else:
            model_name, pretrained_tag = split_model_name_tag(model_name)
            return model_name

    @property
    def pretrained_tag(self) -> Optional[str]:
        if self.pretrained_cfg and 'tag' in self.pretrained_cfg:
            return self.pretrained_cfg['tag']
        elif self.module.pretrained_cfg and 'tag' in self.module.pretrained_cfg:
            return self.module.pretrained_cfg['tag']
        else:
            return None

    @pretrained_tag.setter
    def pretrained_tag(self, new_tag):
        self.pretrained_cfg['tag'] = new_tag

    def get_actual_model_args(self) -> dict:
        model_args = copy.deepcopy(self.model_args)
        try:
            _ = _timm_create_model(model_name=self.model_name, pretrained=False, **model_args)
        except TypeError:
            if 'img_size' in model_args:  # for some model dont support img_size (like mobilenet)
                del model_args['img_size']
                _ = _timm_create_model(model_name=self.model_name, pretrained=False, **model_args)
            else:
                raise

        return model_args

    @classmethod
    def new(cls, model_name: str, tags: List[str], pretrained: bool = True,
            model_args: Optional[dict] = None, pretrained_cfg: Optional[dict] = None):
        model_args = dict(model_args or {})
        pretrained_cfg = dict(pretrained_cfg or {})
        try:
            model = _timm_create_model(model_name=model_name, pretrained=pretrained, **model_args)
        except TypeError:
            if 'img_size' in model_args:  # for some model dont support img_size (like mobilenet)
                _model_args = copy.deepcopy(model_args)
                del _model_args['img_size']
                model = _timm_create_model(model_name=model_name, pretrained=pretrained, **_model_args)
            else:
                raise
        if model.num_classes != len(tags):
            model.reset_classifier(len(tags))
        model.pretrained_cfg.update(pretrained_cfg)

        return cls(
            module=model,
            model_name=model_name,
            tags=tags,
            model_args=model_args,
            pretrained_cfg=pretrained_cfg,
        )

    def save(self, safetensors_file: str, extra_metadata: Dict[str, Any] = None):
        save_model(
            model=self.module,
            filename=safetensors_file,
            metadata={
                **{
                    key: json.dumps(value)
                    for key, value in (extra_metadata or {}).items()
                },
                'model_name': self.model_name,
                'tags': json.dumps(self.tags),
                'model_args': json.dumps(self.model_args),
                'pretrained_cfg': json.dumps(self.pretrained_cfg),
            }
        )

    @classmethod
    def load(cls, safetensors_file: str, device: Union[str, int] = 'cpu', with_metadata: bool = False) \
            -> Union['Model', Tuple['Model', dict]]:
        with safe_open(safetensors_file, 'pt') as f:
            metadata = f.metadata()

        model_name = metadata.pop('model_name')
        tags = json.loads(metadata.pop('tags'))
        model_args = json.loads(metadata.pop('model_args'))
        pretrained_cfg = json.loads(metadata.pop('pretrained_cfg'))
        metadata = {key: json.loads(value) for key, value in metadata.items()}
        model = cls.new(
            model_name=model_name,
            tags=tags,
            model_args=model_args,
            pretrained_cfg=pretrained_cfg,
            pretrained=False,
        )
        load_model(
            model=model.module,
            filename=safetensors_file,
            device=device,
        )

        if with_metadata:
            return model, metadata
        else:
            return model

    def save_as_zip(self, zip_file: str, extra_metadata: Dict[str, Any] = None,
                    train_metadata: Dict[str, Any] = None):
        train_metadata = dict(train_metadata or {})

        with TemporaryDirectory() as td:
            save_datas = {}
            for key, value in train_metadata.items():
                segs = key.split('/')
                if isinstance(value, (Image.Image, pd.DataFrame)):
                    save_datas[key] = value
                else:
                    pkey = '/'.join(segs[:-1])
                    if pkey not in save_datas:
                        save_datas[pkey] = {}
                    save_datas[pkey][segs[-1]] = value

            model_file = os.path.join(td, 'model.safetensors')
            self.save(
                safetensors_file=model_file,
                extra_metadata=extra_metadata,
            )

            for key, value in save_datas.items():
                if isinstance(value, dict):
                    dst_file = os.path.join(td, f'{key}.json')
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    with open(dst_file, 'w') as f:
                        json.dump(value, f, sort_keys=True, ensure_ascii=False)
                elif isinstance(value, Image.Image):
                    dst_file = os.path.join(td, f'{key}.png')
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    value.save(dst_file)
                elif isinstance(value, pd.DataFrame):
                    dst_file = os.path.join(td, f'{key}.csv')
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    value.to_csv(dst_file, index=False)
                else:
                    raise TypeError(f'Unknown value in {key!r} - {value!r}.')

            archive_pack(
                type_name='zip',
                directory=td,
                archive_file=zip_file,
                silent=True,
            )

    @classmethod
    def load_from_zip(cls, zip_file: str, device: Union[str, int] = 'cpu') \
            -> Tuple['Model', dict, dict]:
        with TemporaryDirectory() as td:
            archive_unpack(
                archive_file=zip_file,
                directory=td,
                silent=True,
            )

            model_file = os.path.join(td, 'model.safetensors')
            model, extra_metadata = cls.load(
                safetensors_file=model_file,
                device=device,
                with_metadata=True,
            )

            train_metadata = {}
            for root, _, files in os.walk(td):
                for file in files:
                    dst_file = os.path.abspath(os.path.join(root, file))
                    if os.path.samefile(dst_file, model_file):
                        continue

                    _, ext = os.path.splitext(os.path.basename(dst_file))
                    if ext == '.json':
                        key, _ = os.path.splitext(os.path.relpath(dst_file, td))
                        with open(dst_file, 'r') as f:
                            inner_data = json.load(f)
                        for inner_key, inner_value in inner_data.items():
                            train_metadata[hf_normpath(os.path.join(key, inner_key))] = inner_value
                    elif ext == '.png':
                        key, _ = os.path.splitext(os.path.relpath(dst_file, td))
                        key = hf_normpath(key)
                        image = load_image(dst_file, mode='RGB', force_background='white')
                        image.load()
                        train_metadata[key] = image
                    elif ext == '.csv':
                        key, _ = os.path.splitext(os.path.relpath(dst_file, td))
                        key = hf_normpath(key)
                        df = pd.read_csv(dst_file)
                        train_metadata[key] = df
                    else:
                        logging.warning(f'Unknown file in zip pack {zip_file!r} - {os.path.relpath(dst_file, td)!r}.')

            return model, extra_metadata, train_metadata


if __name__ == '__main__':
    m = Model.new(
        # model_name='caformer_s36.sail_in22k_ft_in1k_384',
        model_name='hf-hub:SmilingWolf/wd-swinv2-tagger-v3',
        tags=['a', 'b', 'c'],
        model_args=dict(drop_path_rate=0.4),
    )
    m.module.eval()

    # quit()
    # print(m)
    print(m.pretrained_tag)
    m.pretrained_tag = 'shit'
    print(m.pretrained_tag)
    print(m.module.pretrained_cfg)
    print(m.architecture)
    print(m.src_repo_id)
    print(m.model_args)
    print(m.pretrained_cfg)
    quit()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        dummy_output = m.module(dummy_input)
        print(dummy_output)
        print(dummy_output.shape)

    m.save('test_safetensors.safetensors', extra_metadata={'train_step': 100})

    mx, mt = Model.load('test_safetensors.safetensors', with_metadata=True)
    mx.module.eval()
    # print(mt)
    print(mx.pretrained_tag)
    print(mx.module.pretrained_cfg)

    with torch.no_grad():
        dummy_output2 = mx.module(dummy_input)
        print(dummy_output2)
        print(dummy_output2.shape)

    torch.testing.assert_close(dummy_output, dummy_output2)
    assert mt == {'train_step': 100}

    df_tags = pd.DataFrame([
        {'tag': '1', 'count': 100},
        {'tag': 'a', 'count': 101},
        {'tag': 'f', 'count': 102},
        {'tag': '4', 'count': 103},
        {'tag': 'l', 'count': 104},
    ])

    m.save_as_zip(
        'test_safetensors.zip',
        extra_metadata={'train_step': 100},
        train_metadata={
            'train/step': 100,
            'train/loss': 1.14514e-2,
            'eval/loss': 1.919810e-2,
            'eval/acc': 0.95,
            'eval/cm': random_bg_image(),
            'eval/tags': df_tags,
        }
    )

    mx, mt, tt = Model.load_from_zip(
        'test_safetensors.zip',
    )
    mx.module.eval()
    print(mx.pretrained_tag)
    print(mx.module.pretrained_cfg)
    with torch.no_grad():
        dummy_output3 = mx.module(dummy_input)
        print(dummy_output3.shape)

    torch.testing.assert_close(dummy_output, dummy_output3)
    assert mt == {'train_step': 100}
    print(tt)
