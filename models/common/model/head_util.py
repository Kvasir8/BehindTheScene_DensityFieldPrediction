from .multi_view_head import MultiViewHead, make_attn_layers

from .resnetfc import ResnetFC
from .mlp import ImplicitNet, make_embedding_encoder


def make_head(conf, d_in: int, d_out: int):
    head_type = conf.get("type", "resnet")

    if head_type == "mlp":
        head = ImplicitNet.from_conf(conf["args"], d_in, d_out)
    elif head_type == "resnet":
        head = ResnetFC.from_conf(conf["args"], d_in, d_out)
    elif head_type == "MultiViewHead":
        head = MultiViewHead.from_conf(conf["args"], d_in, d_out)

    ## For baseline comparison
    # elif head_type == "IBRNet":
    #     head = MultiViewHead.from_conf(conf["args"], d_in, d_out)
    # elif head_type == "NeuRay":
    #     head = MultiViewHead.from_conf(conf["args"], d_in, d_out)
    # elif head_type == "GeoNeRF":
    #     head = MultiViewHead.from_conf(conf["args"], d_in, d_out)
    # elif head_type == "PixelNeRF":
    #     head = MultiViewHead.from_conf(conf["args"], d_in, d_out)
    
    else:
        raise NotImplementedError("Unsupported Head type")
    return head