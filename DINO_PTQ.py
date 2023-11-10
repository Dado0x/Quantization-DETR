import argparse

from datasets.coco import build as build_dataset
from gptq.gptq import *
from gptq.modelutils import *
from gptq.quant import *
from util.build_dino import build_dino_model


@torch.no_grad()
def detr_sequential(args, model, dataloader, dev):
    print('Starting ...')

    layers = torch.nn.ModuleList()

    backbone_idx = -1
    input_projection_idx = -1
    encoder_idx = -1
    decoder_idx = -1
    label_classifier_idx = -1
    bbox_predictor_idx = -1
    enc_output_idx = -1
    enc_out_class_embed_idx = -1
    enc_out_bbox_embed_idx = -1

    # Input encoder
    inps_encoder = [None] * args.nsamples
    inps_pos = [None] * args.nsamples
    inps_reference_points = [None] * args.nsamples
    inps_spatial_shapes = [None] * args.nsamples
    inps_level_start_index = [None] * args.nsamples
    inps_key_padding_mask = [None] * args.nsamples

    # Input decoder
    inps_tgt = [None] * args.nsamples
    inps_tgt_query_pos = [None] * args.nsamples
    inps_tgt_query_sine_embed = [None] * args.nsamples
    inps_tgt_reference_points = [None] * args.nsamples
    inps_memory = [None] * args.nsamples

    # Input backbone
    inps_pixel = [None] * args.nsamples

    # Input output head
    inps_output_head = [None] * args.nsamples

    # Input query selection
    inps_enc_output = [None] * args.nsamples
    inps_enc_out = [None] * args.nsamples

    cache = {'i': 0, "queries": None, "query_position_embeddings": None}

    if args.transformer:
        print('Transformers inputs')
        layers = model.transformer.encoder.layers

        encoder_idx = 0
        decoder_idx = 6

        class CatcherEncoder(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask,
                        **kwargs):
                inps_encoder[cache['i']] = src.cpu()
                inps_pos[cache['i']] = pos.cpu()
                inps_reference_points[cache['i']] = reference_points.cpu()
                inps_spatial_shapes[cache['i']] = spatial_shapes.cpu()
                inps_level_start_index[cache['i']] = level_start_index.cpu()
                inps_key_padding_mask[cache['i']] = key_padding_mask.cpu()
                cache['i'] += 1
                raise ValueError

        ## Encoder
        layers[encoder_idx] = CatcherEncoder(layers[encoder_idx])

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        layers[encoder_idx] = layers[encoder_idx].module

        torch.cuda.empty_cache()

        class CatcherDecoder(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, tgt, **kwargs):
                inps_tgt[cache['i']] = tgt.cpu()
                inps_tgt_query_pos[cache['i']] = kwargs['tgt_query_pos'].cpu()
                inps_tgt_query_sine_embed[cache['i']] = kwargs['tgt_query_sine_embed'].cpu()
                inps_tgt_reference_points[cache['i']] = kwargs['tgt_reference_points'].cpu()
                inps_memory[cache['i']] = kwargs['memory'].cpu()
                cache['i'] += 1
                raise ValueError

        decoder_layers = model.transformer.decoder.layers

        ## Decoder
        decoder_layers[0] = CatcherDecoder(decoder_layers[0])
        cache['i'] = 0

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        decoder_layers[0] = decoder_layers[0].module

        layers = layers + decoder_layers

        torch.cuda.empty_cache()

    if args.backbone:
        backbone_idx = 0
        input_projection_idx = 1
        if args.transformer:
            encoder_idx += 2
            decoder_idx += 2

        print('Backbone inputs')
        cache['i'] = 0

        class CatcherBackbone(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps_pixel[cache['i']] = inp.to("cpu")
                cache['i'] += 1
                raise ValueError

        ##Backbone
        model.backbone = CatcherBackbone(model.backbone)

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        model.backbone = model.backbone.module

        # Add input projection to layers
        layers.insert(0, model.input_proj)

        # Add backbone to layers
        layers.insert(0, model.backbone)

        torch.cuda.empty_cache()

    if args.output_head:
        print('Output head inputs')

        label_classifier_idx = 0
        bbox_predictor_idx = 1

        if args.backbone:
            label_classifier_idx += 2
            bbox_predictor_idx += 2

        if args.transformer:
            label_classifier_idx += 12
            bbox_predictor_idx += 12

        cache['i'] = 0

        class CatcherHead(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps_output_head[cache['i']] = inp.cpu()
                cache['i'] += 1
                raise ValueError

        model.class_embed[-1] = CatcherHead(model.class_embed[-1])

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        model.class_embed[-1] = model.class_embed[-1].module

        layers = layers.append(model.class_embed[-1])
        layers = layers.append(model.bbox_embed[-1])

        torch.cuda.empty_cache()

    if args.query_sel:
        print('Query selection inputs')

        enc_output_idx = 0
        enc_out_class_embed_idx = 1
        enc_out_bbox_embed_idx = 2

        if args.backbone:
            enc_output_idx += 2
            enc_out_class_embed_idx += 2
            enc_out_bbox_embed_idx += 2

        if args.output_head:
            enc_output_idx += 2
            enc_out_class_embed_idx += 2
            enc_out_bbox_embed_idx += 2

        if args.transformer:
            enc_output_idx += 12
            enc_out_class_embed_idx += 12
            enc_out_bbox_embed_idx += 12

        cache['i'] = 0

        query_selection_layer = "enc_output"

        class CatcherQuerySelection(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                if query_selection_layer == "enc_output":
                    inps_enc_output[cache['i']] = inp.cpu()
                elif query_selection_layer == "enc_out_bbox_embed":
                    inps_enc_out[cache['i']] = inp.cpu()
                cache['i'] += 1
                raise ValueError

        model.transformer.enc_output = CatcherQuerySelection(model.transformer.enc_output)

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        model.transformer.enc_output = model.transformer.enc_output.module

        layers = layers.append(model.transformer.enc_output)

        cache['i'] = 0
        query_selection_layer = "enc_out_bbox_embed"

        model.transformer.enc_out_class_embed = CatcherQuerySelection(model.transformer.enc_out_class_embed)

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        model.transformer.enc_out_class_embed = model.transformer.enc_out_class_embed.module

        layers = layers.append(model.transformer.enc_out_class_embed)
        layers = layers.append(model.transformer.enc_out_bbox_embed)

        torch.cuda.empty_cache()

    if args.backbone:
        inps = inps_pixel
    elif args.transformer:
        inps = inps_encoder
    elif args.output_head:
        inps = inps_output_head
    elif args.query_sel:
        inps = inps_enc_output
    else:
        raise ValueError("Can not quantize nothing")

    outs = [None] * args.nsamples

    errors = {}

    model.cpu()

    print('Ready.')
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            print(i, name)
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=False, mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            if i == backbone_idx:  # Backbone
                outs[j] = layer(inps[j].to(dev))[0]
            elif i in [label_classifier_idx, bbox_predictor_idx, enc_output_idx, enc_out_class_embed_idx,
                       enc_out_bbox_embed_idx]:
                layer(inps[j].to(dev))
            elif i == input_projection_idx:  # Input projection
                for l, feat in enumerate(inps[j]):
                    src, _ = feat.decompose()
                    layer[l](src.to(dev))
                layer[-1](inps[j][-1].tensors)
            elif i >= decoder_idx:  # Decoder
                outs[j] = layer(tgt=inps[j].to(dev),
                                tgt_query_pos=inps_tgt_query_pos[j].to(dev),
                                tgt_query_sine_embed=inps_tgt_query_sine_embed[j].to(dev),
                                tgt_reference_points=inps_tgt_reference_points[j].to(dev),
                                memory=inps_memory[j].to(dev),
                                memory_key_padding_mask=inps_key_padding_mask[j].to(dev),
                                memory_level_start_index=inps_level_start_index[j].to(dev),
                                memory_spatial_shapes=inps_spatial_shapes[j].to(dev),
                                memory_pos=inps_pos[j].transpose(0, 1).to(dev))
            else:  # Encoder
                outs[j] = layer(src=inps[j].to(dev), pos=inps_pos[j].to(dev),
                                reference_points=inps_reference_points[j].to(dev),
                                spatial_shapes=inps_spatial_shapes[j].to(dev),
                                level_start_index=inps_level_start_index[j].to(dev))
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            error = gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order,
                static_groups=args.static_groups
            )
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()

            s = ""
            if i == backbone_idx:
                s = f"Backbone_{name[2:]}"
            elif i == input_projection_idx:
                s = f"Input_projection_{name}"
            elif i == label_classifier_idx:
                s = "Label_classifier"
            elif i == bbox_predictor_idx:
                s = f"Bbox_predictor_{name}"
            elif i == enc_output_idx:
                s = "Query_selction_enc_output"
            elif i == enc_out_class_embed_idx:
                s = "Query_selction_enc_out_class_embed"
            elif i == enc_out_bbox_embed_idx:
                s = f"Query_selction_enc_out_bbox_embed_{name[-8:]}"
            elif i >= decoder_idx:
                s = f"Decoder_{i - decoder_idx}_{name}"
            else:
                s = f"Encoder_{i - encoder_idx}_{name}"
            errors[s] = error

        for j in range(args.nsamples):
            if i == backbone_idx:  # Backbone
                outs[j] = layer(inps[j].to(dev))[0]
            elif i in [label_classifier_idx, bbox_predictor_idx, enc_output_idx, enc_out_class_embed_idx,
                       enc_out_bbox_embed_idx]:
                layer(inps[j].to(dev))
            elif i == input_projection_idx:  # Input projection
                for l, feat in enumerate(inps[j]):
                    src, _ = feat.decompose()
                    layer[l](src.to(dev))
                layer[-1](inps[j][-1].tensors)
            elif i >= decoder_idx:  # Decoder
                outs[j] = layer(tgt=inps[j].to(dev),
                                tgt_query_pos=inps_tgt_query_pos[j].to(dev),
                                tgt_query_sine_embed=inps_tgt_query_sine_embed[j].to(dev),
                                tgt_reference_points=inps_tgt_reference_points[j].to(dev),
                                memory=inps_memory[j].to(dev),
                                memory_key_padding_mask=inps_key_padding_mask[j].to(dev),
                                memory_level_start_index=inps_level_start_index[j].to(dev),
                                memory_spatial_shapes=inps_spatial_shapes[j].to(dev),
                                memory_pos=inps_pos[j].transpose(0, 1).to(dev))
            else:  # Encoder
                outs[j] = layer(src=inps[j].to(dev), pos=inps_pos[j].to(dev),
                                reference_points=inps_reference_points[j].to(dev),
                                spatial_shapes=inps_spatial_shapes[j].to(dev),
                                level_start_index=inps_level_start_index[j].to(dev))

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        if i == encoder_idx - 1:  # Encoder inputs
            for k in range(args.nsamples):
                outs[k] = inps_encoder[k].clone()
        if i == decoder_idx - 1:  # Decoder inputs
            for k in range(args.nsamples):
                outs[k] = inps_tgt[k].clone()
        if i == label_classifier_idx:  # Keep decoder outputs for bbox_predictor
            for k in range(args.nsamples):
                outs[k] = inps[k].clone()
        if i == label_classifier_idx - 1: # Label classifier inputs
            for k in range(args.nsamples):
                outs[k] = inps_output_head[k].clone()
        if i == enc_output_idx - 1: # Query selection inputs
            for k in range(args.nsamples):
                outs[k] = inps_enc_output[k].clone()
        if i == enc_out_class_embed_idx - 1: # Query selection inputs
            for k in range(args.nsamples):
                outs[k] = inps_enc_out[k].clone()
        if i == enc_out_class_embed_idx: # Keep query selection outputs for enc_out_bbox_embed
            for k in range(args.nsamples):
                outs[k] = inps[k].clone()

        inps, outs = outs, inps

    print("------------------")
    for k in errors.keys():
        print(k, errors[k])
    print("------------------")
    for k, v in sorted(errors.items(), key=lambda item: -item[1]):
        print(k, v)
    print("------------------")

    name = f"dino_gptq{'_transformer' if args.transformer else ''}{'_backbone' if args.backbone else ''}{'_output_head' if args.output_head else ''}{'_query_selection' if args.query_sel else ''}_{args.nsamples}samples_{args.wbits}bits"

    with open(args.root + name + ".csv", 'w') as f:
        f.write("Layer, Error\n")
        for k, v in errors.items():
            f.write(f"{k}, {v:.5f}\n")

    torch.save(model.state_dict(), args.root + name + ".bin")
    return quantizers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', action='store_true',
                        help='Whether to quantize the backbone. Quantize by default.')

    parser.add_argument('--transformer', action='store_true',
                        help='Whether to quantize the transformer. Quantize by default.')

    parser.add_argument('--output_head', action='store_true',
                        help='Whether to quantize the output head. Quantize by default.')

    parser.add_argument('--query_sel', action='store_true',
                        help='Whether to quantize the query selection. Quantize by default.')

    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling the calibration data.')

    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of calibration data samples.')

    parser.add_argument('--wbits', type=int, default=8, choices=[2, 3, 4, 5, 6, 7, 8],
                        help='#bits to use for quantization; use 16 for evaluating base model.')

    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')

    parser.add_argument('--groupsize', type=int, default=-1,
                        help='Groupsize to use for quantization; default uses full row.')

    parser.add_argument('--act-order', action='store_false',
                        help='Whether to apply the activation order GPTQ heuristic')

    parser.add_argument('--static-groups', action='store_false',
                        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.')

    parser.add_argument('--root', type=str, default='')

    args = parser.parse_args()

    if not args.backbone and not args.transformer and not args.output_head and not args.query_sel:
        args.backbone = True
        args.transformer = True
        args.output_head = True
        args.query_sel = True

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_dino_model(args.root).to(dev)
    model = model.eval()

    print(model)

    dataset_val = build_dataset(image_set='val', coco_path=args.root + "coco")
    dataset_val = torch.utils.data.Subset(dataset_val, torch.arange(0, args.nsamples))
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    dataloader = torch.utils.data.DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False)

    detr_sequential(args, model, dataloader, dev)
