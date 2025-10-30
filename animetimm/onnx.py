import json
import os
from typing import Optional, Dict, Any, Literal

import onnx
import onnxoptimizer
import onnxsim
import torch
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from natsort import natsorted

from .model import Model
from .wrap import wrap_timm_model


def onnx_optimize(model):
    model = onnxoptimizer.optimize(model)
    model, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model


class ExportedONNXNotUniqueError(Exception):
    pass


def export_model_to_onnx(model: Model, dummy_input: torch.Tensor, onnx_filename: str,
                         metadata: Optional[Dict[str, Any]] = None,
                         wrap_mode: Literal['softmax', 'sigmoid'] = 'softmax',
                         verbose: bool = True, opset_version: int = 18, no_optimize: bool = False):
    metadata = dict(metadata or {})
    module = model.module
    module = module.float()

    wrapped_model = wrap_timm_model(module, wrap_mode=wrap_mode)
    dummy_input = dummy_input.float()

    with torch.no_grad(), TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        logging.info(f'Onnx data exporting to {onnx_model_file!r} ...')
        torch.onnx.export(
            wrapped_model,
            (dummy_input,),
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=["embedding", "logits", "prediction"],

            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "embedding": {0: "embedding"},
                "prediction": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
        if os.listdir(td) != [os.path.basename(onnx_model_file)]:
            raise ExportedONNXNotUniqueError(
                'Exported ONNX model expected to be a unique file, '
                f'but {plural_word(len(os.listdir(td)), "file")} found - '
                f'{os.listdir(td)!r}.')

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            logging.info('Optimizing onnx model ...')
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)

        for k, v in natsorted(metadata.items()):
            v = json.dumps(v)
            # logging.info(f'Adding metadata {k!r} = {v!r} ...')
            assert isinstance(v, str)
            meta = model.metadata_props.add()
            meta.key, meta.value = k, v

        logging.info(f'Complete model saving to {onnx_filename!r} ...')
        onnx.save(model, onnx_filename)
