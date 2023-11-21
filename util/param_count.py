from transformers import DetrForObjectDetection

from gptq.modelutils import find_layers
from util.build_dino import build_dino_model


def param_count(layer):
    return sum(p.numel() for p in layer.parameters())


def quant_param_count(layer):
    par_count = 0
    for name, l in find_layers(layer).items():
        par_count += sum(p.numel() for n, p in l.named_parameters() if "bias" not in n)
    return par_count


if __name__ == '__main__':
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to("cpu")

    print("All parameters")
    print("Detr number of parameters : ", param_count(model))
    print("Backbone number of parameters : ", param_count(model.model.backbone))
    print("Input proj number of parameters : ", param_count(model.model.input_projection))
    print("Encoder number of parameters : ", param_count(model.model.encoder))
    print("Decoder number of parameters : ", param_count(model.model.decoder))
    print("Class_labels_classifier number of parameters : ", param_count(model.class_labels_classifier))
    print("Bbox_predictor number of parameters : ", param_count(model.bbox_predictor))

    print()
    print("Quantizable parameters")
    print("Detr number of parameters : ", quant_param_count(model))
    print("Backbone number of parameters : ", quant_param_count(model.model.backbone))
    print("Input proj number of parameters : ", quant_param_count(model.model.input_projection))
    print("Encoder number of parameters : ", quant_param_count(model.model.encoder.layers))
    print("Decoder number of parameters : ", quant_param_count(model.model.decoder.layers))
    print("Class_labels_classifier number of parameters : ", quant_param_count(model.class_labels_classifier))
    print("Bbox_predictor number of parameters : ", quant_param_count(model.bbox_predictor))

    print()
    print("Transformer layer parameters")
    print("Encoder")
    for name, l in find_layers(model.model.encoder.layers[0]).items():
        print(f"{name} number of parameters : ", quant_param_count(l))

    print("Decoder")
    for name, l in find_layers(model.model.decoder.layers[0]).items():
        print(f"{name} number of parameters : ", quant_param_count(l))

    model = build_dino_model("C:/Users/chave/PycharmProjects/Quantization-DETR/").to("cpu")

    print("All parameters")
    print("Dino number of parameters : ", param_count(model))
    print("Backbone number of parameters : ", param_count(model.backbone))
    print("Input proj number of parameters : ", param_count(model.input_proj))
    print("Encoder number of parameters : ", param_count(model.transformer.encoder.layers))
    print("Decoder number of parameters : ", param_count(model.transformer.decoder.layers))
    print("Class_labels_classifier number of parameters : ", param_count(model.class_embed[0]))
    print("Bbox_predictor number of parameters : ", param_count(model.bbox_embed[0]))
    print("Query selection number of parameters : ", param_count(model.transformer.enc_output) +
          param_count(model.transformer.enc_out_bbox_embed) + param_count(model.transformer.enc_out_class_embed))

    print()
    print("Quantizable parameters")
    print("Dino number of parameters : ", quant_param_count(model))
    print("Backbone number of parameters : ", quant_param_count(model.backbone))
    print("Input proj number of parameters : ", quant_param_count(model.input_proj))
    print("Encoder number of parameters : ", quant_param_count(model.transformer.encoder.layers))
    print("Decoder number of parameters : ", quant_param_count(model.transformer.decoder.layers))
    print("Class_labels_classifier number of parameters : ", quant_param_count(model.class_embed[0]))
    print("Bbox_predictor number of parameters : ", quant_param_count(model.bbox_embed[0]))
    print("Query selection number of parameters : ", quant_param_count(model.transformer.enc_output) +
          quant_param_count(model.transformer.enc_out_bbox_embed) + quant_param_count(
        model.transformer.enc_out_class_embed))

    print()
    print("Transformer layer parameters")
    print("Encoder")
    for name, l in find_layers(model.transformer.encoder.layers[0]).items():
        print(f"{name} number of parameters : ", quant_param_count(l))

    print("Decoder")
    for name, l in find_layers(model.transformer.decoder.layers[0]).items():
        print(f"{name} number of parameters : ", quant_param_count(l))

    model = build_dino_model("C:/Users/chave/PycharmProjects/Quantization-DETR/", backbone="swin-L").to("cpu")

    print("All parameters")
    print("Dino number of parameters : ", param_count(model))
    print("Backbone number of parameters : ", param_count(model.backbone))
    print("Input proj number of parameters : ", param_count(model.input_proj))
    print("Encoder number of parameters : ", param_count(model.transformer.encoder.layers))
    print("Decoder number of parameters : ", param_count(model.transformer.decoder.layers))
    print("Class_labels_classifier number of parameters : ", param_count(model.class_embed[0]))
    print("Bbox_predictor number of parameters : ", param_count(model.bbox_embed[0]))
    print("Query selection number of parameters : ", param_count(model.transformer.enc_output) +
          param_count(model.transformer.enc_out_bbox_embed) + param_count(model.transformer.enc_out_class_embed))

    print()
    print("Quantizable parameters")
    print("Dino number of parameters : ", quant_param_count(model))
    print("Backbone number of parameters : ", quant_param_count(model.backbone))
    print("Input proj number of parameters : ", quant_param_count(model.input_proj))
    print("Encoder number of parameters : ", quant_param_count(model.transformer.encoder.layers))
    print("Decoder number of parameters : ", quant_param_count(model.transformer.decoder.layers))
    print("Class_labels_classifier number of parameters : ", quant_param_count(model.class_embed[0]))
    print("Bbox_predictor number of parameters : ", quant_param_count(model.bbox_embed[0]))
    print("Query selection number of parameters : ", quant_param_count(model.transformer.enc_output) +
          quant_param_count(model.transformer.enc_out_bbox_embed) + quant_param_count(
        model.transformer.enc_out_class_embed))
