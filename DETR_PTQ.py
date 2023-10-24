import time

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection

from gptq import *
from modelutils import *
from quant import *
from datasets.coco import build as build_dataset

import argparse

ROOT = "C:/Users/chave/PycharmProjects/Quantization-DETR/"


@torch.no_grad()
def detr_sequential():

    args = argparse.Namespace(nsamples=1000, wbits=4, sym=True, trits=False, percdamp=.01, groupsize=-1, act_order=True, static_groups=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(dev)
    model = model.eval()

    dataset_val = build_dataset(image_set='val', coco_path=ROOT+"coco")
    dataset_val = torch.utils.data.Subset(dataset_val, torch.arange(0, args.nsamples))
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    dataloader = torch.utils.data.DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False)

    print('Starting ...')
    print('Transformers inputs')

    layers = model.model.encoder.layers + model.model.decoder.layers

    inps_encoder = [None] * args.nsamples
    inps_attention_mask = [None] * args.nsamples
    inps_position_embeddings = [None] * args.nsamples
    cache = {'i': 0, "queries": None, "query_position_embeddings": None}

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
            inps_position_embeddings[cache['i']] = kwargs['position_embeddings'].cpu()
            cache['i'] += 1
            raise ValueError

    ## Encoder
    layers[0] = CatcherTransformer(layers[0])
    model.model.encoder.layers[0] = layers[0]

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    model.model.encoder.layers[0] = layers[0].module
    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    ## Decoder
    layers[6] = CatcherTransformer(layers[6])
    model.model.decoder.layers[0] = layers[6]

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    model.model.decoder.layers[0] = layers[6].module
    layers[6] = layers[6].module

    model.cpu()
    torch.cuda.empty_cache()

    model.model.backbone.to(dev)
    model.model.input_projection.to(dev)

    # Add input projection to layers
    layers.insert(0, model.model.input_projection)

    print('Backbone inputs')
    # Add backbone to layers
    layers.insert(0, model.model.backbone)
    cache['i'] = 0
    inps_pixel = [None] * args.nsamples
    inps_pixel_mask = [None] * args.nsamples

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
    layers[0] = CatcherBackbone(layers[0])
    model.model.backbone = layers[0]

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    model.model.backbone = layers[0].module
    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    model.model.backbone.cpu()
    model.model.input_projection.cpu()

    inps = inps_pixel
    outs = [None] * args.nsamples
    inps_encoder_hidden_states = [None] * args.nsamples

    errors = {}
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
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            if i == 0: # Backbone
                outs[j] = layer(inps[j].to(dev), inps_pixel_mask[j].to(dev))[0]
            elif i == 1: # Input projection
                outs[j] = layer(inps[j].to(dev))
            elif i >= 8: # Decoder
                outs[j] = layer(inps[j].to(dev), encoder_hidden_states=inps_encoder_hidden_states[j].to(dev), attention_mask=None, position_embeddings=inps_position_embeddings[j].to(dev), query_position_embeddings=cache['query_position_embeddings'])[0]
            else: # Encoder
                outs[j] = layer(inps[j].to(dev), attention_mask=inps_attention_mask[j].to(dev), position_embeddings=inps_position_embeddings[j].to(dev))[0]
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

            errors[f"{i}_{name}"] = error
        for j in range(args.nsamples):
            if i == 0:  # Backbone
                outs[j] = layer(inps[j].to(dev), inps_pixel_mask[j].to(dev))[0]
            elif i == 1:  # Input projection
                outs[j] = layer(inps[j].to(dev))
            elif i >= 8:  # Decoder
                outs[j] = layer(inps[j].to(dev), encoder_hidden_states=inps_encoder_hidden_states[j].to(dev), attention_mask=None, position_embeddings=inps_position_embeddings[j].to(dev), query_position_embeddings=cache['query_position_embeddings'])[0]
            else: # Encoder
                outs[j] = layer(inps[j].to(dev), attention_mask=inps_attention_mask[j].to(dev), position_embeddings=inps_position_embeddings[j].to(dev))[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        if i == 0: # Outputs backbone
            for k in range(args.nsamples):
                outs[k] = outs[k][0][0]
        if i == 1:
            outs = inps_encoder
        if i == 7: # Outputs encoder
            for k in range(args.nsamples):
                inps_encoder_hidden_states[k] = outs[k].clone()
                outs[k] = cache['queries']

        inps, outs = outs, inps

    print("------------------")
    for k in errors.keys():
        print(k, errors[k])
    print("------------------")
    for k, v in sorted(errors.items(), key=lambda item: -item[1]):
        print(k, v)
    print("------------------")

    torch.save(model.state_dict(), ROOT+f"detr_{args.wbits}bits.bin")
    return quantizers

def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=False)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def load_quant3(model, checkpoint):
    from transformers import OPTConfig, OPTForCausalLM 
    config = OPTConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_quant3(model, layers, faster=False)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

if __name__ == '__main__':
    detr_sequential()

# if __name__ == '__main__':
#     import argparse
#     from datautils import *

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         'model', type=str,
#         help='OPT model to load; pass `facebook/opt-X`.'
#     )
#     parser.add_argument(
#         'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
#         help='Where to extract calibration data from.'
#     )
#     parser.add_argument(
#         '--seed',
#         type=int, default=0, help='Seed for sampling the calibration data.'
#     )
#     parser.add_argument(
#         '--nsamples', type=int, default=128,
#         help='Number of calibration data samples.'
#     )
#     parser.add_argument(
#         '--percdamp', type=float, default=.01,
#         help='Percent of the average Hessian diagonal to use for dampening.'
#     )
#     parser.add_argument(
#         '--nearest', action='store_true',
#         help='Whether to run the RTN baseline.'
#     ) 
#     parser.add_argument(
#         '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
#         help='#bits to use for quantization; use 16 for evaluating base model.'
#     )
#     parser.add_argument(
#         '--trits', action='store_true',
#         help='Whether to use trits for quantization.'
#     )
#     parser.add_argument(
#         '--groupsize', type=int, default=-1,
#         help='Groupsize to use for quantization; default uses full row.'
#     )
#     parser.add_argument(
#         '--sym', action='store_true',
#         help='Whether to perform symmetric quantization.'
#     )
#     parser.add_argument(
#         '--save', type=str, default='',
#         help='Save quantized checkpoint under this name.'
#     )
#     parser.add_argument(
#         '--load', type=str, default='',
#         help='Load quantized model.'
#     )
#     parser.add_argument(
#         '--benchmark', type=int, default=0,
#         help='Number of tokens to use for benchmarking.'
#     )
#     parser.add_argument(
#         '--check', action='store_true',
#         help='Whether to compute perplexity during benchmarking for verification.'
#     )
#     parser.add_argument(
#         '--new-eval', action='store_true',
#         help='Whether to use the new PTB and C4 eval.'
#     )
#     parser.add_argument(
#         '--faster-kernel', action='store_true',
#         help='Whether to use the new faster kernel for benchmarking.'
#     )
#     parser.add_argument(
#         '--act-order', action='store_true',
#         help='Whether to apply the activation order GPTQ heuristic'
#     )
#     parser.add_argument(
#         '--static-groups', action='store_true',
#         help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
#     )

#     args = parser.parse_args()

#     if args.load:
#         model = load_quant3(args.model, args.load)
#     else:
#         model = get_opt(args.model)
#         model.eval()

#     dataloader, testloader = get_loaders(
#         args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
#     )

#     if args.wbits < 16 and not args.nearest:
#         tick = time.time()
#         quantizers = opt_sequential(model, dataloader, DEV)
#         print(time.time() - tick)

#     if args.benchmark:
#         gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
#         if len(gpus) > 1:
#             opt_multigpu(model, gpus)
#         else:
#             model = model.to(DEV)
#         if args.benchmark:
#             input_ids = next(iter(dataloader))[0][:, :args.benchmark]
#             benchmark(model, input_ids, check=args.check)
#     if args.load:
#         exit()

#     datasets = ['wikitext2', 'ptb', 'c4'] 
#     if args.new_eval:
#       datasets = ['wikitext2', 'ptb-new', 'c4-new']
#     for dataset in datasets: 
#         dataloader, testloader = get_loaders(
#             dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
#         )
#         print(dataset)
#         opt_eval(model, testloader, DEV)

#     if args.save:
#         opt_pack3(model, quantizers)
#         torch.save(model.state_dict(), args.save) 
