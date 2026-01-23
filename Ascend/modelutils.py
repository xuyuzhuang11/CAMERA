import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

# DEV = torch.device('cuda:0')
DEV = torch.device('npu:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_moe_module(layer, tp='qwen3'):
    if tp == 'llama':
        assert isinstance(layer, LlamaDecoderLayer)
        return layer.mlp
    elif tp == 'qwen3':
        assert isinstance(layer, Qwen3MoeDecoderLayer)
        return layer.mlp
    elif tp == 'qwen2':
        return layer.mlp
    elif tp == 'd16b':
        # assert isinstance(layer, )
        return layer.mlp
    elif tp == 'phi':
        return layer.block_sparse_moe
    elif tp == 'mix':
        return layer.block_sparse_moe
    elif tp == 'pangu-embedded':
        return layer.mlp
    else:
        raise NotImplementedError(f"Unknown model type: {tp}")
