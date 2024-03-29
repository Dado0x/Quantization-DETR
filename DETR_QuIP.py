import argparse

from tqdm import tqdm
from transformers import DetrForObjectDetection

from QuIP.bal import Balance
from QuIP.gptq import *
from QuIP.near import Nearest
from QuIP.quant import *
from datasets.coco import build as build_dataset
from gptq.modelutils import *


@torch.no_grad()
def detr_sequential(model, dataloader, dev, args):

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
    inps_attention_mask = [None] * args.nsamples
    inps_object_queries = [None] * args.nsamples

    # Input backbone
    inps_pixel = [None] * args.nsamples
    inps_pixel_mask = [None] * args.nsamples

    # Input output head
    inps_output_head = [None] * args.nsamples 

    cache = {'i': 0, "queries": None, "query_position_embeddings": None}

    if args.transformer:
        print('Transformers inputs')
        layers = model.model.encoder.layers + model.model.decoder.layers

        encoder_idx = 0
        decoder_idx = 6

        class CatcherTransformer(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, attention_mask, **kwargs):
                if 'query_position_embeddings' in kwargs.keys():# Decoder
                    cache["queries"] = inp
                    cache['query_position_embeddings'] = kwargs['query_position_embeddings']
                    raise ValueError
                inps_encoder[cache['i']] = inp.cpu()
                inps_attention_mask[cache['i']] = attention_mask.cpu()
                inps_object_queries[cache['i']] = kwargs['object_queries'].cpu()
                cache['i'] += 1
                raise ValueError

        ## Encoder
        layers[encoder_idx] = CatcherTransformer(layers[encoder_idx])
        model.model.encoder.layers[0] = layers[encoder_idx]

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        model.model.encoder.layers[0] = layers[encoder_idx].module
        layers[encoder_idx] = layers[encoder_idx].module

        torch.cuda.empty_cache()

        ## Decoder
        layers[decoder_idx] = CatcherTransformer(layers[decoder_idx])
        model.model.decoder.layers[0] = layers[decoder_idx]

        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

        model.model.decoder.layers[0] = layers[decoder_idx].module
        layers[decoder_idx] = layers[decoder_idx].module

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

    if args.output_head:
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
            print('Output head inputs')
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
    inps_encoder_hidden_states = [None] * args.nsamples

    errors = {}

    model.cpu()

    print('Ready.')

    quantizers = {}
    times = []
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        quant_method = {}
        for name in subset:
            print(i, name)
            # if isinstance(subset[name], nn.Conv2d): ## Conv2d is not supported by ldlq
            #     quant_method[name] = GPTQ(subset[name])
            #     quant_method[name].quantizer = Quantizer()
            #     quant_method[name].quantizer.configure(args.wbits,
            #                                    perchannel=True,
            #                                    sym=False,
            #                                    qfn='a',
            #                                    mse=False)
            if args.quant == 'gptq':
                quant_method[name] = GPTQ(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant == 'near':
                quant_method[name] = Nearest(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant in ['allbal','ldlq','ldlqRG','ldlbal_admm']:
                quant_method[name] = Balance(subset[name])
                quant_method[name].configure(
                                    args.quant,
                                    args.wbits, 
                                    args.npasses,
                                    unbiased=args.unbiased)
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)
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
                outs[j] = layer(inps[j].to(dev), encoder_hidden_states=inps_encoder_hidden_states[j].to(dev),
                                attention_mask=None, object_queries=inps_object_queries[j].to(dev),
                                query_position_embeddings=cache['query_position_embeddings'])[0]
            else: # Encoder
                outs[j] = layer(inps[j].to(dev), attention_mask=inps_attention_mask[j].to(dev),
                                object_queries=inps_object_queries[j].to(dev))[0]
        for h in handles:
            h.remove()

        for name in subset:
            quant_method[name].post_batch()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            # if isinstance(subset[name], nn.Conv2d):
            #     quant_method[name].preproc(preproc_gptqH=True,
            #                                percdamp=args.percdamp,
            #                                preproc_rescale=False,
            #                                preproc_proj=False,
            #                                preproc_proj_extra=0)
            # else:
            quant_method[name].preproc(preproc_gptqH=args.pre_gptqH,
                                       percdamp=args.percdamp,
                                       preproc_rescale=args.pre_rescale,
                                       preproc_proj=args.pre_proj,
                                       preproc_proj_extra=args.pre_proj_extra)
            
            if args.quant == 'gptq':# or isinstance(subset[name], nn.Conv2d):
                quant_method[name].fasterquant(groupsize=args.groupsize)
            elif args.quant in ['allbal','ldlq','ldlqRG','ldlbal_admm']:
                quant_method[name].fasterquant(lazy_batch=args.lazy_batch)
            elif args.quant == 'near':
                quant_method[name].fasterquant()

            times.append(quant_method[name].time)
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = quant_method[name].quantizer
            quant_method[name].free()

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
            errors[s] = quant_method[name].error

        for j in range(args.nsamples):
            if i == backbone_idx:  # Backbone
                outs[j] = layer(inps[j].to(dev), inps_pixel_mask[j].to(dev))[0]
            elif i == input_projection_idx or i == label_classifier_idx or i == bbox_predictor_idx:  # Input projection
                outs[j] = layer(inps[j].to(dev))
            elif i >= decoder_idx:  # Decoder
                outs[j] = layer(inps[j].to(dev), encoder_hidden_states=inps_encoder_hidden_states[j].to(dev),
                                attention_mask=None, object_queries=inps_object_queries[j].to(dev),
                                query_position_embeddings=cache['query_position_embeddings'])[0]
            else: # Encoder
                outs[j] = layer(inps[j].to(dev), attention_mask=inps_attention_mask[j].to(dev),
                                object_queries=inps_object_queries[j].to(dev))[0]

        layers[i] = layer.cpu()
        del layer
        del quant_method
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
                inps_encoder_hidden_states[k] = outs[k].clone()
                outs[k] = cache['queries']
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

    print(f'Total quant time: {sum(times):.2f}s')
    name = f"detr_{args.quant}{'_IP' if args.incoh_processing else ''}{'_unbiased' if args.unbiased else ''}{'_transformer' if args.transformer else ''}{'_backbone' if args.backbone else ''}{'_output_head' if args.output_head else ''}_{args.nsamples}samples_{args.wbits}bits"

    with open(args.root + "errors/" + name + ".csv", 'w') as f:
        f.write("Layer, Error\n")
        for k, v in errors.items():
            f.write(f"{k}, {v:.5f}\n")

    torch.save(model.state_dict(), args.root+"model/"+name+".bin")

    print("Model name : ", name)
    return quantizers, errors


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
    
    parser.add_argument('--nsamples', type=int, default=100,
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

    parser.add_argument('--pre_gptqH', action='store_true',
                        help='preprocessing')
    
    parser.add_argument('--pre_rescale', action='store_true',
                        help='preprocessing')
    
    parser.add_argument('--pre_proj', action='store_true',
                        help='preprocessing')
    
    parser.add_argument('--pre_proj_extra', type=int,default=0, choices=[0, 1, 2],
                        help='Extra options to control pre_proj step.')
    
    parser.add_argument('--qfn', type=str, default='a',
                        help='qfn: a is default, b is sym incoherent based')

    parser.add_argument('--unbiased', action='store_true',
                        help='unbiased')
    
    parser.add_argument('--incoh_processing', action='store_true',
                        help='incoherence processing')
    
    parser.add_argument('--npasses', type=int, default=0,
                        help='number passes to repeat balance loop over 1-d.')
    
    parser.add_argument('--lazy_batch', action='store_true',
                        help='lazy batch updates in blocks as used in OPTQ')
    
    parser.add_argument('--quant', choices=['allbal', 'ldlq', 'ldlqRG', 'ldlbal_admm', 'near', 'gptq'], default='gptq',
                        help='Which quantization method to use.')

    parser.add_argument('--root', type=str)
    
    args = parser.parse_args()

    if not args.backbone and not args.transformer and not args.output_head:
        args.backbone = True
        args.transformer = True
        args.output_head = True

    if args.incoh_processing:
        args.pre_gptqH   = True
        args.pre_rescale = True
        args.pre_proj    = True
        args.proj_extra  = 1
        args.qfn         = 'b'

    if args.qfn=='b': assert args.pre_proj is True
    print(f"Preprocessing flags: gptqH:{args.pre_gptqH}, rescale:{args.pre_rescale}, proj:{args.pre_proj}, proj_extra:{args.pre_proj_extra}, qfn:{args.qfn}")
    print(f"using lazy_batch updates: {args.lazy_batch}")
    # LDL checks
    if ('ldl' in args.quant) and args.unbiased and (args.npasses > 0):
        print(f"LDL NOTE: unbiased + {args.npasses} npasses. NOT TRULY UNBIASED.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(dev)
    model = model.eval()

    dataset_val = build_dataset(image_set='val', coco_path=args.root + "coco")  # Can replace 'val' with 'train' to sample training data
    indices = torch.randperm(len(dataset_val), generator=torch.Generator().manual_seed(args.seed))[:args.nsamples]
    dataset_val = torch.utils.data.Subset(dataset_val, indices)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    dataloader = torch.utils.data.DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    detr_sequential(model, dataloader, dev, args)