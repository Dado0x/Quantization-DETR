import argparse

import torch
import transformers.models.detr.modeling_detr

from dino import build_dino
from util.slconfig import SLConfig


def build_dino_model(root, backbone='resnet50'):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backbone == 'resnet50':
        conf = "DINO_4scale.py"
    elif backbone == 'swin-L':
        conf = "DINO_4scale_swin.py"
    else:
        raise Exception("Unknown backbone")

    args = argparse.Namespace(coco_path=root + "coco",
                              config_file=root + "dino/" + conf,
                              device=dev)

    cfg = SLConfig.fromfile(args.config_file)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)

    model, criterion, postprocessors = build_dino(args)
    if backbone == 'resnet50':
        model.load_state_dict(torch.load(root + "checkpoint0033_4scale.pth", map_location=dev)['model'])
    elif backbone == 'swin-L':
        model.load_state_dict(torch.load(root + "checkpoint0029_4scale_swin.pth", map_location=dev)['model'])
    else:
        raise Exception("Unknown backbone")

    args.decoder_sa_type = 'sa_detr'

    d_model = args.hidden_dim
    n_heads = args.nheads
    dropout = args.dropout

    # replace the decoder self-attention layers with the DETR self-attention layers
    for i, layer in enumerate(model.transformer.decoder.layers):
        self_attn = transformers.models.detr.modeling_detr.DetrAttention(d_model, n_heads, dropout=dropout)

        self_attn.out_proj.weight.data = layer.self_attn.out_proj.weight.data.clone()
        self_attn.out_proj.bias.data = layer.self_attn.out_proj.bias.data.clone()

        q, k, v = torch.split(layer.self_attn.in_proj_weight.data.clone(), d_model, dim=0)
        q_bias, k_bias, v_bias = torch.split(layer.self_attn.in_proj_bias.data.clone(), d_model, dim=0)

        self_attn.q_proj.weight.data = q
        self_attn.q_proj.bias.data = q_bias

        self_attn.k_proj.weight.data = k
        self_attn.k_proj.bias.data = k_bias

        self_attn.v_proj.weight.data = v
        self_attn.v_proj.bias.data = v_bias

        model.transformer.decoder.layers[i].decoder_sa_type = 'sa_detr'
        model.transformer.decoder.layers[i].self_attn = self_attn

    return model