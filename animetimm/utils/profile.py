import torch
from ditk import logging
from thop import profile, clever_format


def torch_model_profile(model, input_):
    with torch.no_grad():
        flops, params = profile(model, (input_,))

    s_flops, s_params = clever_format([flops, params], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_flops}.')

    return flops, params
