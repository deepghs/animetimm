import os.path
from typing import Mapping, Union, Optional

import numpy as np
import torch
import wandb
from PIL import Image
from ditk import logging

from .base import BaseLogger
from ..model import Model


class WandbLogger(BaseLogger):
    def __init__(self, workdir, project="animetimm", name=None, hyperparams: Optional[dict] = None, **kwargs):
        project = project.replace('/', '_').replace('\\', '_').replace('#', '_') \
            .replace(':', '_').replace('?', '_').replace('%', '_')
        BaseLogger.__init__(self, workdir, **kwargs)
        name = name or os.path.basename(workdir)
        self.active = False
        self.wandb_run = None

        try:
            import wandb
            self.wandb = wandb

            if self.wandb.api.api_key is not None:
                self.wandb_run = self.wandb.init(
                    project=project,
                    name=name,
                    dir=workdir,
                    config=dict(hyperparams or {}),
                    # settings=wandb.Settings(console='off'),
                )
                self.active = True
                logging.info("WandbLogger activated successfully.")
            else:
                logging.warning("WandbLogger initialized but no API token found. Running in inactive mode.")
        except (ImportError, Exception) as e:
            logging.warning(f"WandbLogger initialization failed: {str(e)}. Running in inactive mode.")

    def tb_log(self, global_step, data: Mapping[str, Union[float, Image.Image]]):
        if not self.active:
            return

        logging_metrics = {}
        wandb_dict = {}

        for key, value in data.items():
            if isinstance(value, (int, float)) or \
                    (isinstance(value, (torch.Tensor, np.ndarray)) and not value.shape):
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    value = value.tolist()
                wandb_dict[key] = value
                logging_metrics[key] = value
            elif isinstance(value, (Image.Image,)):
                wandb_dict[key] = wandb.Image(value.convert('RGB'), caption=f'#{global_step} - {key}')
            else:
                pass

        if wandb_dict:
            self.wandb_run.log(wandb_dict, step=global_step)

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

    def __del__(self):
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            self.wandb_run.finish()
