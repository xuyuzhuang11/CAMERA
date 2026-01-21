import torch
import torch.nn as nn
import math
import time
import faiss

from typing import List, Dict, Tuple
from collections import defaultdict
import torch.nn.functional as F

import numpy as np


class MEI:
    def __init__(self, module, dev, tp='qwen3', batch_size=128, shape=None):
        self.module = module
        self.dtype = next(iter(module.parameters())).dtype
        self.tp = tp
        self.dev = dev
        self.batch_size = batch_size
        assert shape is not None
        self.n_samples, self.seqlen, self.d_model = shape
        self.n_tokens = self.n_samples * self.seqlen
        self.inp = torch.zeros((self.n_samples, self.seqlen, self.d_model), dtype=self.dtype, device=self.dev)
        self.current_n_samples = 0
    
    def add_batch(self, inp, out):
        self.inp[self.current_n_samples:self.current_n_samples+self.batch_size] = inp
        self.current_n_samples += self.batch_size
        
    def est_importance(self, reduce_ratio=1.0):
        # return importance
        self.inp = self.inp.reshape(-1, self.d_model)
        if self.tp == 'qwen3':
            Gate = self.module.gate.weight
            W_up = torch.stack([expert.up_proj.weight for expert in self.module.experts], dim=0)
            W_gate = torch.stack([expert.gate_proj.weight for expert in self.module.experts], dim=0)
            W_down = torch.stack([expert.down_proj.weight for expert in self.module.experts], dim=0)
            n_experts = Gate.shape[0]
            d_ff = W_up.shape[1]
            
            base_energy = torch.zeros(n_experts*d_ff, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            exp_l2_square = torch.zeros(n_experts*d_ff, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            exp_linf_square = torch.zeros(n_experts*d_ff, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            
            gate_logits = torch.matmul(self.inp, Gate.T)
            gate_scores_all = torch.softmax(gate_logits, dim=-1, dtype=torch.float)
            topn_vals, topn_idx = torch.topk(gate_scores_all, k=8, dim=-1)
            topn_scores = topn_vals / topn_vals.sum(dim=-1, keepdim=True)
            gate_scores = torch.zeros_like(gate_scores_all)
            gate_scores.scatter_(dim=1, index=topn_idx, src=topn_scores).to(self.inp.dtype)
            
            for i in range(n_experts):
                me_start = i * d_ff
                me_end = me_start + d_ff
                down_proj = W_down[i]     # [d_model, d_ff]
                base_energy[me_start:me_end] = (down_proj ** 2).sum(dim=0)  # [d_ff]

                up_proj = W_up[i]
                gate_proj = W_gate[i]
                pre_act = torch.matmul(self.inp, up_proj.T)
                gate = torch.matmul(self.inp, gate_proj.T)
                act_all = torch.nn.functional.silu(gate) * pre_act
                act_all = act_all * gate_scores[:, i].unsqueeze(1)  # [n_tokens, d_ff]
                exp_l2_square[me_start:me_end] += (act_all ** 2).sum(dim=0)  # [d_ff]
                largest_act = torch.max(torch.max(torch.abs(act_all), dim=0)[0] ** 2, exp_linf_square[me_start:me_end])  # [d_ff]
                exp_linf_square[me_start:me_end] = largest_act  # [d_ff]
            
            exp_computed = (1 - reduce_ratio) * exp_l2_square + reduce_ratio * exp_linf_square
            self.importance = base_energy * exp_computed
            self.inp = self.inp.reshape(self.n_samples, self.seqlen, self.d_model)
            return self.importance
        elif self.tp == 'qwen2':
            Gate = self.module.gate.weight
            W_up = torch.stack([expert.up_proj.weight for expert in self.module.experts], dim=0)
            W_gate = torch.stack([expert.gate_proj.weight for expert in self.module.experts], dim=0)
            W_down = torch.stack([expert.down_proj.weight for expert in self.module.experts], dim=0)
            n_experts = Gate.shape[0]
            d_ff = W_up.shape[1]
            
            n_shared_me = self.module.shared_expert.config.shared_expert_intermediate_size
            n_me = n_experts * d_ff + n_shared_me
            base_energy = torch.zeros(n_me, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            exp_l2_square = torch.zeros(n_me, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            exp_linf_square = torch.zeros(n_me, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            
            gate_logits = torch.matmul(self.inp, Gate.T)
            gate_scores_all = torch.softmax(gate_logits, dim=-1, dtype=torch.float)
            topn_vals, topn_idx = torch.topk(gate_scores_all, k=self.module.top_k, dim=-1)
            topn_scores = topn_vals
            gate_scores = torch.zeros_like(gate_scores_all)
            gate_scores.scatter_(dim=1, index=topn_idx, src=topn_scores).to(self.inp.dtype)
            shared_gate_score = F.sigmoid(torch.matmul(self.inp, self.module.shared_expert_gate.weight.T)).reshape(-1, 1)
            
            # out = self.module.shared_experts(self.inp)
            base_energy[:n_shared_me] = (self.module.shared_expert.down_proj.weight ** 2).sum(dim=0)
            pre_act = torch.matmul(self.inp, self.module.shared_expert.up_proj.weight.T)
            gate = torch.matmul(self.inp, self.module.shared_expert.gate_proj.weight.T)
            act_all = torch.nn.functional.silu(gate) * pre_act * shared_gate_score
            exp_l2_square[:n_shared_me] += (act_all ** 2).sum(dim=0)
            largest_act = torch.max(torch.max(torch.abs(act_all), dim=0)[0] ** 2, exp_linf_square[:n_shared_me])
            exp_linf_square[:n_shared_me] = largest_act
            for i in range(n_experts):
                me_start = n_shared_me + i * d_ff
                me_end = me_start + d_ff
                down_proj = W_down[i]     # [d_model, d_ff]
                base_energy[me_start:me_end] = (down_proj ** 2).sum(dim=0)  # [d_ff]

                up_proj = W_up[i]
                gate_proj = W_gate[i]
                pre_act = torch.matmul(self.inp, up_proj.T)
                gate = torch.matmul(self.inp, gate_proj.T)
                act_all = torch.nn.functional.silu(gate) * pre_act
                act_all = act_all * gate_scores[:, i].unsqueeze(1)  # [n_tokens, d_ff]
                exp_l2_square[me_start:me_end] += (act_all ** 2).sum(dim=0)  # [d_ff]
                largest_act = torch.max(torch.max(torch.abs(act_all), dim=0)[0] ** 2, exp_linf_square[me_start:me_end])  # [d_ff]
                exp_linf_square[me_start:me_end] = largest_act  # [d_ff]
            
            exp_computed = (1 - reduce_ratio) * exp_l2_square + reduce_ratio * exp_linf_square
            self.importance = base_energy * exp_computed
            self.inp = self.inp.reshape(self.n_samples, self.seqlen, self.d_model)
            return self.importance
        elif self.tp == 'mix' or self.tp == 'phi':
            Gate = self.module.gate.weight
            W_up = torch.stack([expert.w3.weight for expert in self.module.experts], dim=0)
            W_gate = torch.stack([expert.w1.weight for expert in self.module.experts], dim=0)
            W_down = torch.stack([expert.w2.weight for expert in self.module.experts], dim=0)
            n_experts = Gate.shape[0]
            d_ff = W_up.shape[1]
            
            base_energy = torch.zeros(n_experts*d_ff, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            exp_l2_square = torch.zeros(n_experts*d_ff, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            exp_linf_square = torch.zeros(n_experts*d_ff, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            
            # gate_logits = torch.matmul(self.inp, Gate.T)
            # gate_scores = torch.softmax(gate_logits, dim=-1, dtype=torch.float).to(self.inp.dtype)        # [n_tokens, n_experts]
            gate_logits = torch.matmul(self.inp, Gate.T)
            gate_scores_all = torch.softmax(gate_logits, dim=-1, dtype=torch.float)
            top2_vals, top2_idx = torch.topk(gate_scores_all, k=2, dim=-1)
            top2_scores = top2_vals / top2_vals.sum(dim=-1, keepdim=True)
            gate_scores = torch.zeros_like(gate_scores_all)
            gate_scores.scatter_(dim=1, index=top2_idx, src=top2_scores).to(self.inp.dtype)
            
            for i in range(n_experts):
                me_start = i * d_ff
                me_end = me_start + d_ff
                down_proj = W_down[i]     # [d_model, d_ff]
                base_energy[me_start:me_end] = (down_proj ** 2).sum(dim=0)  # [d_ff]

                up_proj = W_up[i]
                gate_proj = W_gate[i]
                pre_act = torch.matmul(self.inp, up_proj.T)
                gate = torch.matmul(self.inp, gate_proj.T)
                act_all = torch.nn.functional.silu(gate) * pre_act
                act_all = act_all * gate_scores[:, i].unsqueeze(1)  # [n_tokens, d_ff]
                exp_l2_square[me_start:me_end] += (act_all ** 2).sum(dim=0)  # [d_ff]
                largest_act = torch.max(torch.max(torch.abs(act_all), dim=0)[0] ** 2, exp_linf_square[me_start:me_end])  # [d_ff]
                exp_linf_square[me_start:me_end] = largest_act  # [d_ff]
            
            exp_computed = (1 - reduce_ratio) * exp_l2_square + reduce_ratio * exp_linf_square
            self.importance = base_energy * exp_computed
            self.inp = self.inp.reshape(self.n_samples, self.seqlen, self.d_model)
            return self.importance
        elif self.tp == 'd16b':
            Gate = self.module.gate.weight
            W_up = torch.stack([expert.up_proj.weight for expert in self.module.experts], dim=0)
            W_gate = torch.stack([expert.gate_proj.weight for expert in self.module.experts], dim=0)
            W_down = torch.stack([expert.down_proj.weight for expert in self.module.experts], dim=0)
            n_experts = Gate.shape[0]
            d_ff = W_up.shape[1]
            
            n_shared_me = self.module.config.n_shared_experts * self.module.config.moe_intermediate_size
            n_me = n_experts * d_ff + n_shared_me
            base_energy = torch.zeros(n_me, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            exp_l2_square = torch.zeros(n_me, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            exp_linf_square = torch.zeros(n_me, device=self.dev, dtype=self.dtype)  # [n_experts * d_ff]
            
            gate_logits = torch.matmul(self.inp, Gate.T)
            gate_scores_all = torch.softmax(gate_logits, dim=-1, dtype=torch.float)
            topn_vals, topn_idx = torch.topk(gate_scores_all, k=self.module.config.num_experts_per_tok, dim=-1)
            topn_scores = topn_vals / topn_vals.sum(dim=-1, keepdim=True)
            gate_scores = torch.zeros_like(gate_scores_all)
            gate_scores.scatter_(dim=1, index=topn_idx, src=topn_scores).to(self.inp.dtype)
            
            # out = self.module.shared_experts(self.inp)
            base_energy[:n_shared_me] = (self.module.shared_experts.down_proj.weight ** 2).sum(dim=0)
            pre_act = torch.matmul(self.inp, self.module.shared_experts.up_proj.weight.T)
            gate = torch.matmul(self.inp, self.module.shared_experts.gate_proj.weight.T)
            act_all = torch.nn.functional.silu(gate) * pre_act
            exp_l2_square[:n_shared_me] += (act_all ** 2).sum(dim=0)
            largest_act = torch.max(torch.max(torch.abs(act_all), dim=0)[0] ** 2, exp_linf_square[:n_shared_me])
            exp_linf_square[:n_shared_me] = largest_act
            for i in range(n_experts):
                me_start = n_shared_me + i * d_ff
                me_end = me_start + d_ff
                down_proj = W_down[i]     # [d_model, d_ff]
                base_energy[me_start:me_end] = (down_proj ** 2).sum(dim=0)  # [d_ff]

                up_proj = W_up[i]
                gate_proj = W_gate[i]
                pre_act = torch.matmul(self.inp, up_proj.T)
                gate = torch.matmul(self.inp, gate_proj.T)
                act_all = torch.nn.functional.silu(gate) * pre_act
                act_all = act_all * gate_scores[:, i].unsqueeze(1)  # [n_tokens, d_ff]
                exp_l2_square[me_start:me_end] += (act_all ** 2).sum(dim=0)  # [d_ff]
                largest_act = torch.max(torch.max(torch.abs(act_all), dim=0)[0] ** 2, exp_linf_square[me_start:me_end])  # [d_ff]
                exp_linf_square[me_start:me_end] = largest_act  # [d_ff]
            
            exp_computed = (1 - reduce_ratio) * exp_l2_square + reduce_ratio * exp_linf_square
            self.importance = base_energy * exp_computed
            self.inp = self.inp.reshape(self.n_samples, self.seqlen, self.d_model)
            return self.importance
        else:
            raise NotImplementedError(f"Unknown model type: {self.tp}")

    def make_pruned_module(self, ratio):
        # make pruned_module
        if self.tp == 'qwen3':
            n_experts = self.module.gate.weight.shape[0]
            d_ff = self.module.experts[0].up_proj.weight.shape[0]
            n_me = n_experts * d_ff
            n_prune = int(n_me * ratio)
            _, indices = torch.topk(self.importance, k=n_prune, largest=False)
            expert_ids = indices // d_ff
            channel_ids = indices % d_ff
            for e, c in zip(expert_ids.tolist(), channel_ids.tolist()):
                self.module.experts[e].up_proj.weight.data[c, :] = 0.0
                self.module.experts[e].gate_proj.weight.data[c, :] = 0.0
                self.module.experts[e].down_proj.weight.data[:, c] = 0.0
        elif self.tp == 'qwen2':
            n_experts = self.module.gate.weight.shape[0]
            d_ff = self.module.experts[0].up_proj.weight.shape[0]
            n_shared_me = self.module.shared_expert.config.shared_expert_intermediate_size
            n_me = n_experts * d_ff + n_shared_me
            n_prune = int(n_me * ratio)
            _, indices = torch.topk(self.importance, k=n_prune, largest=False)
            expert_ids = indices // d_ff
            channel_ids = indices % d_ff
            shared_channel_ids = []
            for e, c in zip(expert_ids.tolist(), channel_ids.tolist()):
                if e >= 8:
                    ee = e - 8
                    self.module.experts[ee].up_proj.weight.data[c, :] = 0.0
                    self.module.experts[ee].gate_proj.weight.data[c, :] = 0.0
                    self.module.experts[ee].down_proj.weight.data[:, c] = 0.0
                else:
                    shared_channel_ids.append(e * d_ff + c)
            self.module.shared_expert.up_proj.weight.data[shared_channel_ids, :] = 0.0
            self.module.shared_expert.gate_proj.weight.data[shared_channel_ids, :] = 0.0
            self.module.shared_expert.down_proj.weight.data[:, shared_channel_ids] = 0.0
        elif self.tp == 'phi' or self.tp == 'mix':
            n_experts = self.module.gate.weight.shape[0]
            d_ff = self.module.experts[0].w3.weight.shape[0]
            n_me = n_experts * d_ff
            n_prune = int(n_me * ratio)
            _, indices = torch.topk(self.importance, k=n_prune, largest=False)
            # all_indices = torch.arange(n_me, device=self.dev, dtype=torch.long)
            # mask = ~torch.isin(all_indices, self.indices)
            # indices = all_indices[mask]
            expert_ids = indices // d_ff
            channel_ids = indices % d_ff
            for e, c in zip(expert_ids.tolist(), channel_ids.tolist()):
                self.module.experts[e].w3.weight.data[c, :] = 0.0
                self.module.experts[e].w1.weight.data[c, :] = 0.0
                self.module.experts[e].w2.weight.data[:, c] = 0.0
        elif self.tp == 'd16b':
            n_experts = self.module.gate.weight.shape[0]
            d_ff = self.module.experts[0].up_proj.weight.shape[0]
            n_shared_me = self.module.config.n_shared_experts * self.module.config.moe_intermediate_size
            n_me = n_experts * d_ff + n_shared_me
            n_prune = int(n_me * ratio)
            _, indices = torch.topk(self.importance, k=n_prune, largest=False)
            expert_ids = indices // d_ff
            channel_ids = indices % d_ff
            shared_channel_ids = []
            for e, c in zip(expert_ids.tolist(), channel_ids.tolist()):
                if e >= self.module.config.n_shared_experts:
                    ee = e - self.module.config.n_shared_experts
                    self.module.experts[ee].up_proj.weight.data[c, :] = 0.0
                    self.module.experts[ee].gate_proj.weight.data[c, :] = 0.0
                    self.module.experts[ee].down_proj.weight.data[:, c] = 0.0
                else:
                    shared_channel_ids.append(e * d_ff + c)
            self.module.shared_experts.up_proj.weight.data[shared_channel_ids, :] = 0.0
            self.module.shared_experts.gate_proj.weight.data[shared_channel_ids, :] = 0.0
            self.module.shared_experts.down_proj.weight.data[:, shared_channel_ids] = 0.0
        else:
            raise NotImplementedError(f"Unknown model type: {self.tp}")
