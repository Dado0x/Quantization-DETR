import torch
import torchvision
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from datasets.coco_eval import CocoEvaluator
from tqdm import tqdm


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, feature_extractor):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

ROOT = "C:/Users/chave/PycharmProjects/Quantization-DETR/"

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)
    model.load_state_dict(torch.load(ROOT + "detr_4bits.bin"))
    model.eval()

    feature_extractor = DetrFeatureExtractor()

    dataset_val = CocoDetection(img_folder=ROOT + 'coco/val2017',
                                ann_file=ROOT + 'coco/annotations/instances_val2017.json',
                                feature_extractor=feature_extractor)
    dataloader = torch.utils.data.DataLoader(dataset_val, 2, collate_fn=collate_fn)

    base_ds = dataset_val.coco
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    print("Running evaluation...")

    for idx, batch in enumerate(tqdm(dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in
                  batch["labels"]]  # these are in DETR format, resized + normalized

        # forward pass
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = feature_extractor.post_process(outputs, orig_target_sizes)  # convert outputs to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    print(coco_evaluator.summarize())
