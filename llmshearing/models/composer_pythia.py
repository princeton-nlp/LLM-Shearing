import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from torch.nn import functional as F
from transformers.pytorch_utils import (find_pruneable_heads_and_indices,
                                        prune_linear_layer)

from llmshearing.models.l0_module import L0Module
from llmshearing.models.composer_llama import ComposerMosaicLlama, prepare_decoder_attention_mask, turn_head_z, turn_mlp_z, normal_attn_fn, flash_attn_fn
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

class ComposerMosaicPythia(ComposerMosaicLlama):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = PythiaModel(cfg)
        

class CoFiLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device)

    def forward(self, input, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(
                input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = input.clone()
            normed_input = normed_input.to(output.dtype) 
            output[..., remaining_index] = normed_input
        else:
            output = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output
    
    def prune_params(self, hidden_z):
        remaining_index = torch.where(~hidden_z.eq(0))[0]
        # self.weight = torch.nn.Parameter(self.weight.data.mul(hidden_z.squeeze())[remaining_index])
        self.weight = torch.nn.parameter.Parameter(self.weight.index_select(0, remaining_index))
        self.bias = torch.nn.parameter.Parameter(self.bias.index_select(0, remaining_index))
        self.normalized_shape = (len(remaining_index),)

class PythiaEmbedding(nn.Embedding):
    def forward(self, input, hidden_z=None):
        embeddings = super().forward(input)
        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
        return embeddings

    def prune_params(self, hidden_z):
        remaining_index = torch.where(~hidden_z.eq(0))[0]
        self.weight.data = self.weight.data.mul(hidden_z)
        self.weight = torch.nn.parameter.Parameter(self.weight.index_select(1, remaining_index).clone())
        self.embedding_dim = len(remaining_index)
        print(f"    Embedding: {len(hidden_z)} -> {len(remaining_index)}")
        
    
class PythiaModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        print(f'Tried to build Pythia model with cfg.name={cfg.name}')
        self.cfg = cfg
        
        ### added ###
        self.l0_module = None
        if getattr(self.cfg, "l0_module", None) is not None:
            self.l0_module = L0Module(self.cfg, device=cfg.init_device)
        #############

        layernorm_class = CoFiLayerNorm
        self.attn_impl = cfg.attn_impl

        self.embedding_fraction = cfg.get('embedding_fraction', 1)
        assert 0 < self.embedding_fraction <= 1, 'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'

    
        self.transformer = nn.ModuleDict({
            "wte": PythiaEmbedding(cfg.vocab_size, 
                                   cfg.d_model,
                                   device=cfg.init_device),
        })
        
        self.transformer.update({
            'blocks':
                nn.ModuleList([
                    PythiaBlock(cfg, device=cfg.init_device)
                    for _ in range(cfg.n_layers)
                ])
        })
        self.transformer.update({
            "output": nn.Linear(cfg.d_model, cfg.vocab_size, device=cfg.init_device, bias=False),
        })
        
        self.transformer.update({
            "ln_f": layernorm_class(cfg.d_model, eps=cfg.layer_norm_eps, device=cfg.init_device), # TODO: add to config
        })
        
        self.is_causal = True 
        if cfg.get('verbose') and cfg.get('verbose') > 2:
            print(self)
        
    def prune_params(self, zs=None):
        # TODO 
        if zs is None:
            self.l0_module.eval()
            zs = self.l0_module(calculate_lagrangian=False)
        # wte as well :) 
        # ln_f if hidden states are to be pruned
        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"]
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            self.transformer.ln_f.prune_params(hidden_z)
            self.transformer.wte.weight.data = self.transformer.wte.weight.data.mul(hidden_z)
            self.transformer.wte.weight = torch.nn.parameter.Parameter(
                self.transformer.wte.weight.index_select(1, remaining_index).clone())
            self.transformer.wte.embedding_dim = len(remaining_index)
            # self.transformer.output.weight.data = self.transformer.output.weight.data.mul(hidden_z) 
            half = self.transformer.output.weight.data.dtype == torch.float16
            self.transformer.output = prune_linear_layer(self.transformer.output, remaining_index, dim=1)
            if half:
                self.transformer.output = self.transformer.output.half()
            
        for i, block in enumerate(self.transformer.blocks):
            zs_block = self.get_zs_block(zs, i)
            block.prune_params(zs_block)
        
    def get_zs_block(self, zs, block_idx):
        zs_block = {}
        if zs is not None:
            for key in zs:
                if key == "hidden_z": zs_block["hidden_z"] = zs["hidden_z"]
                else: zs_block[key] = zs[key][block_idx] 
        return zs_block

    def forward(
            self,
            input_ids: torch.LongTensor,
            key_padding_mask: Optional[torch.ByteTensor] = None,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            pruned_steps: int = 0,
            retain_grad: bool = False,
            **zs,):
        S = input_ids.size(1)
        assert S <= self.cfg.max_seq_len, f"Sequence length ({S}) exceeds model maximum sequence length ({self.cfg.max_seq_len})!"
        
        tok_emb = self.transformer.wte(input_ids)
        if "hidden_z" in zs:
            tok_emb = tok_emb.mul(zs["hidden_z"])
        
        x = tok_emb 
        
        attn_bias = None # only consider the flash attention case
        attention_mask = prepare_decoder_attention_mask((tok_emb.size(0), tok_emb.size(1)), tok_emb)
        
        l0_output = None
        if self.l0_module is not None:
            assert zs == {}, "zs should be empty when using L0Module"
            zs = self.l0_module(calculate_lagrangian=False, pruned_steps=pruned_steps)
            
        for b_idx, block in enumerate(self.transformer.blocks):
            zs_block = self.get_zs_block(zs, b_idx)
            past_key_value = past_key_values[
                b_idx] if past_key_values is not None else None

            x, past_key_value = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                key_padding_mask=key_padding_mask,
                is_causal=self.is_causal,
                attention_mask=attention_mask,
                retain_grad=retain_grad,
                **zs_block 
            )

            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value

        x = self.transformer.ln_f(x, hidden_z=zs.get("hidden_z", None))
        logits = self.transformer.output(x)

        if self.l0_module is not None:
            l0_output = self.l0_module(calculate_lagrangian=True, pruned_steps=pruned_steps)

        return {"logits": logits, "l0_output": l0_output, "zs": zs}
        
    def param_init_fn(self, module):
        pass
        
    def fsdp_wrap_fn(self, module):
        return isinstance(module, PythiaBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, PythiaBlock)
    

class PythiaBlock(nn.Module):
    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()

        layernorm_class = CoFiLayerNorm # TODO: CoFiLayerNorm,RMSLayerNorm
        
        self.ln_1 = layernorm_class(cfg.d_model, eps=cfg.layer_norm_eps, device=device)
        self.attn = PythiaAttention(cfg, device) 
        self.ln_2 = layernorm_class(cfg.d_model, eps=cfg.layer_norm_eps, device=device)
        self.mlp = PythiaMLP(cfg, device)
        
        self.use_parallel_residual = cfg.get('use_parallel_residual', False) # TODO: add to config
    
    def prune_params(self, zs_block):
        self.attn.prune_params(zs_block)
        self.mlp.prune_params(zs_block)
        
        if self.attn.query_key_value is None:
            self.ln_1 = None
        if self.mlp.up_proj is None:
            self.ln_2 = None
        if "hidden_z" in zs_block:
            hidden_z = zs_block["hidden_z"]
            if self.ln_1 is not None: self.ln_1.prune_params(hidden_z)
            if self.ln_2 is not None: self.ln_2.prune_params(hidden_z) 
        
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        retain_grad: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
        qk_head_dim_z: Optional[torch.Tensor] = None,
        vo_head_dim_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        if self.ln_1 is not None:
            a = self.ln_1(x, hidden_z=hidden_z)
            attn_output, _, past_key_value = self.attn(a, 
                                             past_key_value=past_key_value,
                                             attn_bias=attn_bias,
                                             key_padding_mask=key_padding_mask,
                                             is_causal=is_causal,
                                             attention_mask=attention_mask,
                                             retain_grad=retain_grad,
                                             head_z=head_z,
                                             head_layer_z=head_layer_z,
                                             hidden_z=hidden_z,
                                             qk_head_dim_z=qk_head_dim_z,
                                             vo_head_dim_z=vo_head_dim_z)
        else:
            attn_output = 0
            
        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            if self.ln_2 is not None:
                b = self.ln_2(x, hidden_z=hidden_z)
                mlp_output = self.mlp(b, retain_grad, intermediate_z, mlp_z, hidden_z)
                x = mlp_output + attn_output + x 
            else:
                x = attn_output + x
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            if self.ln_2 is not None:
                attn_output = x + attn_output
                hidden_states = self.ln_2(attn_output, hidden_z=hidden_z)
                mlp_output = self.mlp(hidden_states, retain_grad, intermediate_z, mlp_z, hidden_z)
                x = mlp_output + attn_output 
            else:
                x = x + attn_output
        return x, past_key_value 

    
class PythiaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.attn_impl = cfg.get('attn_impl')
        
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.all_head_size = cfg.d_model
        self.head_dim = self.d_model // self.n_heads 
        self.pruned_heads = set()
        
        self.softmax_scale = cfg.get('softmax_scale')
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = cfg.get('attn_pdrop')
        
        # self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device, bias=False)
        # for param init fn; enables shape based init of fused layers
        # fuse_splits = (cfg.d_model, 2 * cfg.d_model)
        # self.Wqkv._fused = (0, fuse_splits)  # type: ignore
        self.query_key_value = nn.Linear(self.d_model, 3 * self.d_model, device=device, bias=True)
        fuse_splits = (cfg.d_model, 2 * cfg.d_model)
        self.query_key_value._fused = (0, fuse_splits)
        
        self.attn_fn = flash_attn_fn if self.attn_impl == 'flash' else normal_attn_fn

        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device, bias=True)
        self.out_proj._is_residual = True  # type: ignore
        
        self.rotary_ndims = int(self.head_dim * cfg.rotary_pct) 
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims, max_position_embeddings=cfg.max_seq_len, device=device)
    
    def prune_params(self, zs_block):
        head_z = None; head_layer_z = None; hidden_z = None; qk_head_dim_z = None; vo_head_dim_z = None
        if "head_z" in zs_block:
            head_z = zs_block["head_z"].squeeze()
        
        if "head_layer_z" in zs_block:
            head_layer_z = zs_block["head_layer_z"].squeeze()
        
        if "hidden_z" in zs_block:
            hidden_z = zs_block["hidden_z"].squeeze()
        
        # update params #
        if head_z is not None:
            head_z_for_update = torch.repeat_interleave(head_z, self.head_dim)
            start_index = torch.arange(0, self.n_heads * 3, 3) + 2
            end_index = start_index + 1
            index = torch.cat([torch.arange(i, j) for i, j in zip(start_index * self.head_dim, end_index * self.head_dim)])
            self.query_key_value.weight.data[index, :] = \
                self.query_key_value.weight.data.transpose(0, 1)[:, index].mul(head_z_for_update).transpose(0, 1)
            self.query_key_value.bias.data[index] = \
                self.query_key_value.bias.data[index].mul(head_z_for_update)
        if head_layer_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
            self.out_proj.bias.data = self.out_proj.bias.data.mul(head_layer_z)
        if hidden_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
            self.out_proj.bias.data = self.out_proj.bias.data.mul(hidden_z)
        #################
        
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            print(f"    Head hidden: {len(hidden_z)} -> {len(remaining_index)}") 
            half = next(self.query_key_value.parameters()).dtype == torch.float16
            self.query_key_value = prune_linear_layer(self.query_key_value, remaining_index, dim=1)
            self.out_proj = prune_linear_layer(self.out_proj, remaining_index)
            if half:
                self.query_key_value.half()
                self.out_proj.half()
         
        to_prune_heads = turn_head_z(head_z, head_layer_z)
        len_to_prune_heads = len(to_prune_heads)
        if len_to_prune_heads == 0:
            print(f"    Heads: {self.n_heads} -> {self.n_heads}")
            return

        heads, index = find_pruneable_heads_and_indices(
            to_prune_heads, self.n_heads, self.head_dim, self.pruned_heads
        )
        
        # Prune linear layers
        # setting layers to be None if all the heads are pruned
        if len(index) == 0:
            self.query_key_value = None
            self.out_proj = None
        else:
            half = next(self.query_key_value.parameters()).dtype == torch.float16
            remaining_heads = list(set([i for i in range(self.n_heads)]) - set(to_prune_heads))
            qkv_index = torch.cat([torch.arange(i * self.head_dim * 3, (i+1) * self.head_dim * 3).to(index.device) for i in remaining_heads])
            self.query_key_value = prune_linear_layer(self.query_key_value, qkv_index) 
            self.out_proj = prune_linear_layer(self.out_proj, index, dim=1)
            if half:
                self.query_key_value.half()
                self.out_proj.half()

        print(f"    Heads: {self.n_heads} -> {self.n_heads - len(heads)}")

        # Update hyper params and store pruned heads
        self.n_heads = self.n_heads - len(heads)
        self.all_head_size = self.head_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
            
    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        key_padding_mask=None,
        is_causal=True,
        needs_weights=False,
        attention_mask=None,
        retain_grad=False,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
        qk_head_dim_z=None,
        vo_head_dim_z=None):

        if self.query_key_value is None:
            return None, None, past_key_value

        qkv = self.query_key_value(x)
        
        query_padding_mask = None
        if key_padding_mask is not None:
            query_padding_mask = key_padding_mask[:, -query.size(1):]
        
        # b, s, d = query.shape 
        new_qkv_shape = qkv.size()[:-1] + (self.n_heads, 3 * self.head_dim)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_dim].permute(0, 2, 1, 3)
        key = qkv[..., self.head_dim : 2 * self.head_dim].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_dim :].permute(0, 2, 1, 3)
        
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]
        
        kv_seq_len = key.size(2)
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)

        position_ids = torch.arange(offset, kv_seq_len, dtype=torch.long, device=cos.device)
        position_ids = position_ids.unsqueeze(0).view(-1, kv_seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        offset = 0
        if past_key_value is not None:
            if len(past_key_value) != 0:
                offset = past_key_value[0].shape[-2]
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)
                past_key_value = (key, value)

        if self.attn_fn == flash_attn_fn: # TODO: test if it is the same as attn
            query = rearrange(query, 'b h s d -> b s h d')
            key = rearrange(key, 'b h s d -> b s h d')
            value = rearrange(value, 'b h s d -> b s h d')
            context, attn_weights = self.attn_fn(
                query,
                key,
                value,
                self.n_heads,
                softmax_scale=self.softmax_scale,
                attn_bias=attn_bias,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                dropout_p=self.attn_dropout_p,
                training=self.training,
                needs_weights=needs_weights,
                head_z=head_z
            )
        else:
            context = self.attn_fn(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                head_z=head_z
            )
            attn_weights = None
        if retain_grad:
            self.context = context
            if self.context.requires_grad:
                self.context.retain_grad()
        output = self.out_proj(context)
        
        if head_layer_z is not None:
            output *= head_layer_z
        
        if hidden_z is not None:
            output *= hidden_z
            
        if retain_grad: 
            self.output = output 
            if self.output.requires_grad:
                self.output.retain_grad()
        
        return output, attn_weights, past_key_value


class PythiaMLP(nn.Module):
    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=True, device=device)
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=True, device=device)

    def prune_params(self, zs_block):
        intermediate_z = zs_block.get("intermediate_z", None)
        mlp_z = zs_block.get("mlp_z", None)
        hidden_z = zs_block.get("hidden_z", None)
        # update params #
        if intermediate_z is not None:
            self.down_proj.weight.data = self.down_proj.weight.data.mul(intermediate_z.squeeze(0))
        if mlp_z is not None:
            self.down_proj.weight.data = self.down_proj.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
            self.down_proj.bias.data = self.down_proj.bias.data.mul(mlp_z)
        if hidden_z is not None:
            self.down_proj.weight.data = self.down_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1) 
            self.down_proj.bias.data = self.down_proj.bias.data.mul(hidden_z)
        #################

        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            print(f"    FFN hidden dim: {len(hidden_z)} -> {len(remaining_index)}")
            half = next(self.up_proj.parameters()).dtype
            self.up_proj = prune_linear_layer(self.up_proj, remaining_index, dim=1)
            self.down_proj = prune_linear_layer(self.down_proj, remaining_index, dim=0)
            if half == torch.float16:
                self.up_proj = self.up_proj.half()
                self.down_proj = self.down_proj.half()
            
        keep_dim = turn_mlp_z(intermediate_z, mlp_z)
        device = self.up_proj.weight.device
        if len(keep_dim) == self.up_proj.weight.shape[0]:
            print(f"    FFN intermediate dim: {self.cfg.intermediate_size} -> {len(keep_dim)}")
            return 
            
        if len(keep_dim) == 0:
            self.up_proj = None; self.down_proj = None
        else:
            keep_dim_index = torch.tensor(keep_dim).long().to(device)
            half = next(self.up_proj.parameters()).dtype
            self.up_proj = prune_linear_layer(self.up_proj, keep_dim_index, dim=0)
            self.down_proj = prune_linear_layer(self.down_proj, keep_dim_index, dim=1)
            if half == torch.float16:
                self.up_proj = self.up_proj.half()
                self.down_proj = self.down_proj.half()
        print(f"    FFN intermediate dim: {self.cfg.intermediate_size} -> {len(keep_dim)}")
        
    def forward(self, x, retain_grad=False, intermediate_z=None, mlp_z=None, hidden_z=None):
        if self.up_proj is None:
            return None
        up_v = F.gelu(self.up_proj(x))
        if retain_grad:
            self.up_v = up_v
            if self.up_v.requires_grad:
                self.up_v.retain_grad()
        if intermediate_z is not None:    
            up_v *= intermediate_z
        down_v = self.down_proj(up_v)
        
        if retain_grad:
            self.output = down_v
            if self.output.requires_grad:
                self.output.retain_grad()
        
        if mlp_z is not None:
            down_v = down_v * mlp_z
            
        if hidden_z is not None:
            down_v = down_v * hidden_z
            
        return down_v
    

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)
