from typing import List, Type, Optional, Mapping, Union

from PIL import Image

from .base import BaseLogger
from .checkpoint import CheckpointLogger
from .tensorboard import TensorboardLogger
from .wandb import WandbLogger
from ..model import Model

_DEFAULT_CLASSES: List[Type[BaseLogger]] = [TensorboardLogger, CheckpointLogger, WandbLogger]


class TrainSession:
    def __init__(self, workdir: str, classes: Optional[List[Type[BaseLogger]]] = None, **kwargs):
        if classes is None:
            classes = _DEFAULT_CLASSES
        self.loggers = [cls(workdir, **kwargs) for cls in classes]

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        for logger in self.loggers:
            logger.tb_train_log(global_step, metrics)

    def tb_eval_log(self, global_step, model: Model, metrics: Mapping[str, Union[float, Image.Image]]):
        for logger in self.loggers:
            logger.tb_eval_log(global_step, model, metrics)
