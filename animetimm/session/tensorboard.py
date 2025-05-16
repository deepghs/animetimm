import io
from typing import Mapping, Union

import numpy as np
import torch
from PIL import Image
from ditk import logging
from torch.utils.tensorboard import SummaryWriter

from .base import BaseLogger
from ..model import Model


class TensorboardLogger(BaseLogger):
    def __init__(self, workdir, **kwargs):
        BaseLogger.__init__(self, workdir, **kwargs)
        self.tb_writer = SummaryWriter(workdir)

    def tb_log(self, global_step, data: Mapping[str, Union[float, Image.Image]]):
        logging_metrics = {}
        for key, value in data.items():
            if isinstance(value, (int, float)) or \
                    (isinstance(value, (torch.Tensor, np.ndarray)) and not value.shape):
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    value = value.tolist()
                self.tb_writer.add_scalar(key, value, global_step)
                logging_metrics[key] = value
            elif isinstance(value, (Image.Image,)):
                # noinspection PyTypeChecker
                img_array = np.asarray(value.convert('RGB')).transpose((2, 0, 1))
                self.tb_writer.add_image(key, img_array, global_step, dataformats='CHW')
            else:
                pass

        if logging_metrics:
            with io.StringIO() as sf:
                print(f'Metrics logging at epoch {global_step}', file=sf, end='')
                for key, value in logging_metrics.items():
                    print(f', {key}: {value:.4g}', file=sf, end='')

                logging.info(sf.getvalue())

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        self.tb_log(
            global_step,
            {f'train/{key}': value for key, value in metrics.items()}
        )

    def tb_eval_log(self, global_step, model: Model, metrics: Mapping[str, Union[float, Image.Image]]):
        _ = model
        self.tb_log(
            global_step,
            {f'eval/{key}': value for key, value in metrics.items()}
        )
