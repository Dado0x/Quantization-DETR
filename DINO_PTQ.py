import argparse

from transformers import DeformableDetrForObjectDetection

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
    inps_pixel_mask = [None] * args.nsamples

    # Input output head
    inps_output_head = [None] * args.nsamples 

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
            def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask, **kwargs):
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
        model.model.backbone.to(dev)
        model.model.input_projection.to(dev)

        backbone_idx = 0
        input_projection_idx = 1
        if args.transformer:
            encoder_idx += 2
            decoder_idx += 2

        # Add input projection to layers
        layers.insert(0, model.model.input_projection)

        print('Backbone inputs')
        # Add backbone to layers
        layers.insert(0, model.model.backbone)
        cache['i'] = 0

        class CatcherBackbone(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, pixel_values, pixel_mask, **kwargs):
                inps_pixel[cache['i']] = pixel_values.cpu()
                inps_pixel_mask[cache['i']] = pixel_mask.cpu()
                cache['i'] += 1
                raise ValueError

        ##Backbone
        layers[backbone_idx] = CatcherBackbone(layers[backbone_idx])
        model.model.backbone = layers[backbone_idx]

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        model.model.backbone = layers[backbone_idx].module
        layers[backbone_idx] = layers[backbone_idx].module

        torch.cuda.empty_cache()

        model.model.backbone.cpu()
        model.model.input_projection.cpu()

    if args.output_head:
        print('Output head inputs')
        layers = layers.append(model.class_labels_classifier)
        layers = layers.append(model.bbox_predictor)

        label_classifier_idx = 0
        bbox_predictor_idx = 1

        if args.backbone:
            label_classifier_idx += 2
            bbox_predictor_idx += 2

        if args.transformer:
            label_classifier_idx += 12
            bbox_predictor_idx += 12
        
        if not args.transformer:
            cache['i'] = 0
            class CatcherHead(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps_output_head[cache['i']] = inp.cpu()
                    cache['i'] += 1
                    raise ValueError

            layers[label_classifier_idx] = CatcherHead(layers[label_classifier_idx])
            model.class_labels_classifier = layers[label_classifier_idx]

            for batch in dataloader:
                try:
                    model(batch[0].to(dev))
                except ValueError:
                    pass

            model.class_labels_classifier = layers[label_classifier_idx].module
            layers[label_classifier_idx] = layers[label_classifier_idx].module

            torch.cuda.empty_cache()

    if args.backbone:
        inps = inps_pixel
    elif args.transformer:
        inps = inps_encoder
    elif args.output_head:
        inps = inps_output_head 
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
            if i == backbone_idx: # Backbone
                outs[j] = layer(inps[j].to(dev), inps_pixel_mask[j].to(dev))[0]
            elif i == input_projection_idx or i == label_classifier_idx or i == bbox_predictor_idx:  # Input projection
                outs[j] = layer(inps[j].to(dev))
            elif i >= decoder_idx: # Decoder
                outs[j] = layer(tgt=inps[j].to(dev),
                                tgt_query_pos=inps_tgt_query_pos[j].to(dev),
                                tgt_query_sine_embed=inps_tgt_query_sine_embed[j].to(dev),
                                tgt_reference_points=inps_tgt_reference_points[j].to(dev),
                                memory=inps_memory[j].to(dev),
                                memory_key_padding_mask=inps_key_padding_mask[j].to(dev),
                                memory_level_start_index=inps_level_start_index[j].to(dev),
                                memory_spatial_shapes=inps_spatial_shapes[j].to(dev),
                                memory_pos=inps_pos[j].transpose(0, 1).to(dev))
            else: # Encoder
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
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()

            s = ""
            if i == backbone_idx:
                s = f"Backbone_{name}"
            elif i == input_projection_idx:
                s = "Input_projection"
            elif i == label_classifier_idx:
                s = "Label_classifier"
            elif i == bbox_predictor_idx:
                s = f"Bbox_predictor_{name}"
            elif i >= decoder_idx:
                s = f"Decoder_{i-decoder_idx}_{name}"
            else:
                s = f"Encoder_{i-encoder_idx}_{name}"
            errors[s] = error

        for j in range(args.nsamples):
            if i == backbone_idx:  # Backbone
                outs[j] = layer(inps[j].to(dev), inps_pixel_mask[j].to(dev))[0]
            elif i == input_projection_idx or i == label_classifier_idx or i == bbox_predictor_idx:  # Input projection
                outs[j] = layer(inps[j].to(dev))
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
            else: # Encoder
                outs[j] = layer(src=inps[j].to(dev), pos=inps_pos[j].to(dev),
                                reference_points=inps_reference_points[j].to(dev),
                                spatial_shapes=inps_spatial_shapes[j].to(dev),
                                level_start_index=inps_level_start_index[j].to(dev))

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        if i == backbone_idx: # Keep backbone outputs
            for k in range(args.nsamples):
                outs[k] = outs[k][0][0]
        if i == input_projection_idx:
            if args.transformer:
                outs = inps_encoder # Encoder inputs
            else:
                outs = inps_output_head # Output_head inputs
        if i == decoder_idx-1: # Decoder inputs
            for k in range(args.nsamples):
                outs[k] = inps_tgt[k].clone()
        if i == label_classifier_idx: # Keep decoder outputs for bbox_predictor
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

    name = f"dino_gptq{'_transformer' if args.transformer else ''}{'_backbone' if args.backbone else ''}{'_output_head' if args.output_head else ''}_{args.nsamples}samples_{args.wbits}bits"

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

    parser.add_argument('--seed',type=int, default=0, 
                        help='Seed for sampling the calibration data.')
    
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of calibration data samples.')

    parser.add_argument('--wbits', type=int, default=8, choices=[2, 3, 4, 5, 6, 7, 8],
                        help='#bits to use for quantization; use 16 for evaluating base model.')
    
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    
    parser.add_argument('--groupsize',type=int,default=-1,
                        help='Groupsize to use for quantization; default uses full row.')
    
    parser.add_argument('--act-order', action='store_false',
                        help='Whether to apply the activation order GPTQ heuristic')
    
    parser.add_argument('--static-groups', action='store_false', 
                        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.')

    parser.add_argument('--root', type=str, default='')

    args = parser.parse_args()

    if not args.backbone and not args.transformer and not args.output_head:
        args.backbone = True
        args.transformer = True
        args.output_head = True

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(args.root)

    if args.root != "":
        ROOT = args.root

    model = build_dino_model(args.root)
    model.load_state_dict(torch.load(args.root + "checkpoint0033_4scale.pth", map_location=dev)['model'])
    model = model.eval()

    print(model)

    #transformers.models.detr.modeling_detr.DetrAttention(d_model, n_heads, dropout=dropout)

    dataset_val = build_dataset(image_set='val', coco_path=ROOT+"coco")
    dataset_val = torch.utils.data.Subset(dataset_val, torch.arange(0, args.nsamples))
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    dataloader = torch.utils.data.DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False)

    detr_sequential(args, model, dataloader, dev)
