import argparse

import torch
from tqdm import tqdm

from datasets.coco import build
from datasets.coco_eval import CocoEvaluator
from dino.dino import PostProcess
from util.build_dino import build_dino_model
from util.misc import collate_fn


def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    else:
        raise NotImplementedError("Call Shilong if you use other containers! type: {}".format(type(item)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--coco', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--root', type=str)

    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_dino_model(args.root).to(dev)
    #model.load_state_dict(torch.load("C:/Users/chave/PycharmProjects/Quantization-DETR/checkpoint0033_4scale.pth", map_location=dev)['model'])

    print(model)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    dataset_val = build("val", args.coco)
    dataloader = torch.utils.data.DataLoader(dataset_val, 1, collate_fn=collate_fn)

    base_ds = dataset_val.coco
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    postprocessors = {'bbox': PostProcess(num_select=300, nms_iou_threshold=-1)}

    print("Running evaluation...")

    for samples, targets in tqdm(dataloader):
        samples = samples.to(dev)

        targets = [{k: to_device(v, dev) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    print(coco_evaluator.summarize())
