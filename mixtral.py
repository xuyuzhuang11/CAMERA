import time
import torch

from modelutils import *
from mei import MEI
# from minigptq import MiniGPTQ


def get_mixtral(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, config=config, torch_dtype='auto')          # load in cpu
    model.seqlen = 2048
    return model


@torch.no_grad()
def mix_importance(model, dataloader, dev):
    print('Starting ...')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, args.calib_length, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']:cache['i']+args.chunk_size] = inp
            cache['i'] += args.chunk_size
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs['position_embeddings']
            cache['cache_position'] = kwargs['cache_position']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(0, len(dataloader), args.chunk_size):
        batch_group = dataloader[i:i+args.chunk_size]
        input_batch = torch.concat([x[0] for x in batch_group], dim=0)
        try:
            model(input_batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    cache_position = cache['cache_position']
    print('Ready.')
    
    importances = []
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        # if i >= model.config.first_k_dense_replace:
        module = get_moe_module(layer, 'mix')
        mei = MEI(module, dev, tp='mix', batch_size=args.chunk_size, shape=inps.shape)
        # mei.importance = torch.load(args.save + '/importance.pt')[i-model.config.first_k_dense_replace]
        
        def _add_batch():
            def tmp(_, inp, out):
                mei.add_batch(inp[0].data, out[0].data)
            return tmp
        handle = module.register_forward_hook(_add_batch())
        for j in range(0, args.nsamples, args.chunk_size):
            outs[j:j+args.chunk_size] = layer(inps[j:j+args.chunk_size], attention_mask=attention_mask, position_ids=position_ids, 
                                              position_embeddings=position_embeddings, cache_position=cache_position)[0]
        handle.remove()
    
        print(f'get importance vector of {i}-th layer')
        importances.append(mei.est_importance(args.reduce_ratio))
        if args.func == 'prune':
            print(f'make pruned module of {i}-th layer with ratio {args.pratio}')
            mei.make_pruned_module(ratio=args.pratio)
        print('done.')
        
        for j in range(0, args.nsamples, args.chunk_size):
            outs[j:j+args.chunk_size] = layer(inps[j:j+args.chunk_size], attention_mask=attention_mask, position_ids=position_ids,
                                              position_embeddings=position_embeddings, cache_position=cache_position)[0]
        layers[i] = layer.cpu()
        del layer
        # if i >= model.config.first_k_dense_replace:
        del mei
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    model.config.use_cache = use_cache
    print('All done!')
    return importances


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Phi model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'c4',],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=1234, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--calib-length', type=int, default=2048,
        help='Length of calibration data samples.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save importance order and model dict in this path.'
    )
    parser.add_argument(
        '--pratio', type=float, default=0.2,
        help='Ratio of structral pruning.'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=128,
        help='Chunk size of parallel computing.'
    )
    parser.add_argument(
        '--func', type=str, default='prune',
        help='Functions of this script to run. [prune]'
    )
    parser.add_argument(
        '--wbits', type=int, nargs='+', default=[8, 4, 2],
        help='Bit-widths of quantization for each group. [group1, group2, group3]'
    )
    parser.add_argument(
        '--bits-ratio', type=float, nargs='+', default=[0.1, 0.3, 0.6],
        help='Ratio of each group. [group1, group2, group3]'
    )
    parser.add_argument(
        '--reduce_ratio', type=float, default=1.0,
        help='Ratio of top reducing tokens.'
    )
    args = parser.parse_args()
    
    model = get_mixtral(args.model)
    model.eval()
    print("model loaded!")

    if args.func == 'prune':
        dataloader, _ = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.calib_length
        )
        tick = time.time()
        importances = mix_importance(model, dataloader, DEV)
        print(f"time: {time.time() - tick}")
    
    if args.save:
        if args.func == 'prune':
            torch.save(importances, args.save + '/importance.pt')
            torch.save(model.state_dict(), args.save + '/prune_model_dict.pt')
            # checkpoint = torch.load(args.save + '/model_dict.pt')
            # model.load_state_dict(checkpoint)
