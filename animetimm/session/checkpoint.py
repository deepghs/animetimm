import os
from typing import Optional, Mapping, Union, Dict, Any

import pandas as pd
from PIL import Image
from ditk import logging

from .base import BaseLogger
from ..model import Model


class CheckpointLogger(BaseLogger):
    __GLOBAL_METADATA__ = {
        'copyright': 'DeepGHS (https://github.com/deepghs)',
        'project': 'AnimeTIMM (https://huggingface.co/animetimm)',
        'homepage': 'https://huggingface.co/animetimm',
        'source': 'https://github.com/deepghs/animetimm',
    }

    def __init__(self, workdir: str, key_metric: str = 'accuracy',
                 extra_metadata: Optional[Dict[str, Any]] = None, save_step_when_better: bool = False, **kwargs):
        BaseLogger.__init__(self, workdir, **kwargs)
        self.key_metric = key_metric
        self.extra_metadata = dict(extra_metadata or {})

        self.ckpt_dir = os.path.join(self.workdir, 'checkpoints')
        self._last_step: Optional[int] = None
        self._best_metric_value: Optional[float] = None
        self._save_step_when_better = save_step_when_better
        self._load_last()
        self._load_best()

    @classmethod
    def _load_zip_ckpt(cls, zip_file: str):
        _, meta, metrics = Model.load_from_zip(zip_file)
        return meta, metrics

    def _save_zip_ckpt(self, zip_file: str, global_step, model: Model,
                       metrics: Mapping[str, Union[float, Image.Image]]):
        if os.path.dirname(zip_file):
            os.makedirs(os.path.dirname(zip_file), exist_ok=True)
        model.save_as_zip(
            zip_file=zip_file,
            extra_metadata={
                **self.__GLOBAL_METADATA__,
                **self.extra_metadata,
                **{
                    f'metrics/{key}': value for key, value in metrics.items()
                    if not isinstance(value, (Image.Image, pd.DataFrame))
                },
                'step': global_step,
            },
            train_metadata={
                **{
                    f'metrics/{key}': value for key, value in metrics.items()
                    if not isinstance(value, (Image.Image, pd.DataFrame))
                },
                'metrics/step': global_step,
                **{
                    f'plot_{key}': value for key, value in metrics.items()
                    if isinstance(value, Image.Image)
                },
                **{
                    f'df_{key}': value for key, value in metrics.items()
                    if isinstance(value, pd.DataFrame)
                },
            }
        )

    @property
    def _last_zip_ckpt(self):
        return os.path.join(self.ckpt_dir, 'last.zip')

    def _step_zip_ckpt(self, step: int):
        return os.path.join(self.ckpt_dir, f'step-{step:06d}.zip')

    def _load_last(self):
        if os.path.exists(self._last_zip_ckpt):
            meta, metrics = self._load_zip_ckpt(self._last_zip_ckpt)
            self._last_step = meta['step']
            logging.info(f'Last ckpt found at {self._last_step}, with previous step {self._last_step}')
        else:
            self._last_step = None
            logging.info('No last ckpt found.')

    def _save_last(self, global_step, model: Model, metrics: Mapping[str, Union[float, Image.Image]]):
        self._save_zip_ckpt(
            zip_file=self._last_zip_ckpt,
            global_step=global_step,
            model=model,
            metrics=metrics,
        )
        self._last_step = global_step
        logging.info(f'Last ckpt model epoch {global_step} saved')

    @property
    def _best_zip_ckpt(self):
        return os.path.join(self.ckpt_dir, 'best.zip')

    def _load_best(self):
        if os.path.exists(self._best_zip_ckpt):
            meta, metrics = self._load_zip_ckpt(self._best_zip_ckpt)
            self._best_metric_value = metrics[f'metrics/{self.key_metric}']
            step = meta['step']
            logging.info(f'Best ckpt found at {self._best_zip_ckpt}, '
                         f'with step {step} and {self.key_metric} {self._best_metric_value:.3f}')
        else:
            self._best_metric_value = None
            logging.info('No best ckpt found.')

    def _save_best(self, global_step, model: Model, metrics: Mapping[str, Union[float, Image.Image]]):
        if self._best_metric_value is None or metrics[self.key_metric] > self._best_metric_value:
            self._save_zip_ckpt(
                zip_file=self._best_zip_ckpt,
                global_step=global_step,
                model=model,
                metrics=metrics,
            )
            if self._save_step_when_better:
                self._save_zip_ckpt(
                    zip_file=self._step_zip_ckpt(global_step),
                    global_step=global_step,
                    model=model,
                    metrics=metrics,
                )
            self._best_metric_value = metrics[self.key_metric]
            logging.info(f'Best ckpt model epoch {global_step} saved, '
                         f'with {self.key_metric}\'s new value {self._best_metric_value:.3f}')

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        pass

    def tb_eval_log(self, global_step, model: Model, metrics: Mapping[str, Union[float, Image.Image]]):
        self._save_last(global_step, model, metrics)
        self._save_best(global_step, model, metrics)
