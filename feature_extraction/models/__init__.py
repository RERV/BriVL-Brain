from .vl_model import VL_model

import torch

__all__ = {
    'VL': VL_model
}


def build_network(model_cfg=None, args=None):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg,
        args=args
    )
    return model
