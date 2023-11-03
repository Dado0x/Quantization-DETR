import argparse

from dino import build_dino
from util.slconfig import SLConfig


def build_dino_model(root):

    args = argparse.Namespace(coco_path=root + "coco",
                              config_file=root + "dino/DINO_4scale.py",
                              device="cpu")

    cfg = SLConfig.fromfile(args.config_file)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)

    model, criterion, postprocessors = build_dino(args)

    return model