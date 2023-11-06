import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from composer.metrics import METRIC_DEFAULT_CTORS
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from composer.models.base import ComposerModel
from composer.utils import dist, get_device, reproducibility
from einops import rearrange
from omegaconf import DictConfig
from torch.nn import functional as F
from transformers.pytorch_utils import (find_pruneable_heads_and_indices,
                                        prune_linear_layer)

from llmshearing.models.l0_module import L0Module
from llmshearing.models.metrics import DomainCount, DomainLanguageCrossEntropy


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device: Optional[str] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.variance_epsilon = eps
    
    def prune_params(self, hidden_z):
        remaining_index = torch.where(~hidden_z.eq(0))[0]
        self.weight = torch.nn.Parameter(self.weight.data.mul(hidden_z.squeeze())[remaining_index])

    def forward(self, hidden_states, hidden_z=None): 
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(hidden_states, dim=-1, index=remaining_index)
        else:
            compressed_input = hidden_states
        variance = compressed_input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        output = self.weight * hidden_states
        if hidden_z is not None:
            output = output.mul(hidden_z)
        return output

class ComposerMosaicLlama(ComposerModel):
    """ Llama model with the Composer model interface. """
    def __init__(self, cfg):
        super().__init__()
        self.model = LlamaModel(cfg)
        self.ref_model = None
        self.num_fwd_flops = self._compute_num_fwd_flops()
        self.train_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(),
            'Perplexity': LanguagePerplexity(),
        }
        self.eval_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(),
            'Perplexity': LanguagePerplexity(),
        }

        self.set_names = getattr(cfg, "set_names", None)
        if self.set_names is not None:
            self.set_name_to_id = {set_name: i for i, set_name in enumerate(self.set_names)}
            self.set_id_to_name = {i: set_name for i, set_name in enumerate(self.set_names)}
        
            for set_name in self.set_names:
                # add train and eval metrics for each set
                self.train_metrics[f'{set_name}_LanguageCrossEntropy'] = DomainLanguageCrossEntropy(set_name=set_name)
                self.eval_metrics[f'{set_name}_LanguageCrossEntropy'] = DomainLanguageCrossEntropy(set_name=set_name)
                self.train_metrics[f'{set_name}_count'] = DomainCount(set_name=set_name, set_index=self.set_name_to_id[set_name]) 

    def prune_params(self, zs=None):
        self.model.prune_params(zs)
        
    def get_targets(self, batch):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets
        
    def forward(self, batch):
        input_ids = batch['input_ids']
        key_padding_mask = batch['attention_mask'].bool(
        ) if 'attention_mask' in batch else None
        pruned_steps = batch.get('pruned_steps', None)
        if pruned_steps is not None:
            pruned_steps = pruned_steps[0].item()
        zs = {key: batch[key] for key in batch if "_z" in key}
        model_output = self.model(input_ids=input_ids, key_padding_mask=key_padding_mask, pruned_steps=pruned_steps, **zs)
        return model_output

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        logits = outputs["logits"]
        l0_output = outputs["l0_output"]
        targets = self.get_targets(batch)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-100)
        return_loss = {"ce_loss": loss}
        if l0_output is not None:
            lag_loss = l0_output[0]
            return_loss["lag_loss"] = lag_loss
        return_loss["total"] = sum(return_loss.values())
        return return_loss

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric) -> None:
        logits = outputs["logits"]
        if isinstance(metric, DomainLanguageCrossEntropy):
            targets = self.get_targets(batch)
            set_id = self.set_name_to_id[metric.set_name]
            targets[batch["set"] != set_id] = -100
            metric.update(logits, targets)
        elif isinstance(metric, DomainCount):
            with torch.inference_mode():
                idx = None
                selected_sets = batch['set']
            metric.update(selected_sets, idx)
        else:
            logits = logits.view(-1, logits.size(-1))
            targets = self.get_targets(batch).view(-1)
            metric.update(logits, targets)

    def add_eval_metrics(self, evaluator):
        evaluator_metrics = {
            m: METRIC_DEFAULT_CTORS[m]() for m in evaluator.metric_names
        }
        if self.eval_metrics is not None:
            self.eval_metrics.update(evaluator_metrics)
        else:
            self.eval_metrics = evaluator_metrics

    def _compute_num_fwd_flops(self):
        # Might not be correct for LLaMA structures
        n_params = sum(p.numel() for p in self.parameters())
        # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.model.cfg.max_seq_len
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = self.model.cfg.n_layers * 2 * 2 * (
            self.model.cfg.d_model * (self.model.cfg.max_seq_len**2))
        return params_flops_per_seq + attn_flops_per_seq

    def flops_per_batch(self, batch):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass
        return self.num_fwd_flops * 3 * batch['input_ids'].shape[0]

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        if new_num_tokens is not None:
            self.model._resize_token_embeddings(new_num_tokens)
    
    
class LlamaModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        print(f'Tried to build Llama model with cfg.name={cfg.name}')
        self.cfg = cfg
        
        ### added ###
        self.l0_module = None
        if getattr(self.cfg, "l0_module", None) is not None:
            self.l0_module = L0Module(self.cfg, device=cfg.init_device)
        #############

        layernorm_class = LlamaRMSNorm # TODO: CoFiLayerNorm,RMSLayerNorm
        self.attn_impl = cfg.attn_impl

        self.embedding_fraction = cfg.get('embedding_fraction', 1)
        assert 0 < self.embedding_fraction <= 1, 'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, 
                                cfg.d_model,
                                device=cfg.init_device),
        })
        self.transformer.update({
            'blocks':
                nn.ModuleList([
                    LlamaBlock(cfg, device=cfg.init_device)
                    for _ in range(cfg.n_layers)
                ])
        })
        self.transformer.update({
            "ln_f": layernorm_class(cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=cfg.init_device),
        })
        self.transformer.update({
            "output": nn.Linear(cfg.d_model, cfg.vocab_size, device=cfg.init_device, bias=False),
        })
        
        self.is_causal = True 
        
        # define attn mask
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = None

        if cfg.get('verbose') and cfg.get('verbose') > 2:
            print(self)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.transformer.wte
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, nn.Embedding)
        self.transformer.wte = new_embeddings

        old_lm_head = self.transformer.output
        new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens, nn.Linear)
        self.transformer.output = new_lm_head

        self.cfg.vocab_size = new_num_tokens
    
    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None, new_type=nn.Embedding
    ) -> nn.Embedding:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return 

        # Build new embeddings
        if new_type == nn.Embedding:
            new_embeddings = new_type(new_num_tokens, old_embedding_dim)
        else:
            new_embeddings = new_type(old_embedding_dim, new_num_tokens, bias=False)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        input_embeddings_avg = old_embeddings.weight.mean(dim=0, keepdim=True)
        new_embeddings.weight.data[n:] = input_embeddings_avg

        return new_embeddings
        
        
    def prune_params(self, zs=None):
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
            # This is def a bug in llama, but does not incur too much issue
            self.transformer.output.weight.data = self.transformer.output.weight.data.mul(hidden_z) 
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
        
    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module):
        pass
        # init_fn_name = self.cfg.get('param_init_fn', 'baseline_')
        # if self.cfg.get('verbose', 0) > 1:
        #     warnings.warn(f'Using {init_fn_name} initialization.')
        # MODEL_INIT_REGISTRY[init_fn_name](module, self.cfg)
        
    def fsdp_wrap_fn(self, module):
        return isinstance(module, LlamaBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, LlamaBlock)

class LlamaBlock(nn.Module):
    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()

        layernorm_class = LlamaRMSNorm # TODO: CoFiLayerNorm,RMSLayerNorm
        
        self.ln_1 = layernorm_class(cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=device)
        self.attn = LlamaAttention(cfg, device) 
        self.ln_2 = layernorm_class(cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=device)
        self.mlp = LlamaMLP(cfg, device)
    
    def prune_params(self, zs_block):
        self.attn.prune_params(zs_block)
        self.mlp.prune_params(zs_block)
        # ln_1, ln_2 later
        
        if self.attn.wq is None:
            self.ln_1 = None
        if self.mlp.gate_proj is None:
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
            b, _, past_key_value = self.attn(a, 
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
            b = 0
        x = x + b
        
        if self.ln_2 is not None:
            m = self.ln_2(x, hidden_z=hidden_z)
            n = self.mlp(m, retain_grad, intermediate_z, mlp_z, hidden_z)
        else:
            n = 0
             
        x = x + n        
        return x, past_key_value 
    
def turn_head_z(head_z, head_layer_z):
    head_z = head_z.squeeze().clone()
    if head_layer_z is not None:
        head_z *= head_layer_z
    to_prune_heads = torch.where(head_z == 0)[0].view(-1).tolist()
    return to_prune_heads

def turn_mlp_z(intermediate_z, mlp_z):
    intermediate_z_layer = intermediate_z.squeeze().clone()
    if mlp_z is not None:
        intermediate_z_layer *= mlp_z
    keep_intermediate_dims = torch.where(intermediate_z_layer != 0)[0].tolist()
    return keep_intermediate_dims 


class LlamaAttention(nn.Module):
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
        self.wq = nn.Linear(self.d_model, self.d_model, device=device, bias=False)
        self.wk = nn.Linear(self.d_model, self.d_model, device=device, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_model, device=device, bias=False)
        
        self.attn_fn = flash_attn_fn if self.attn_impl == 'flash' else normal_attn_fn

        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device, bias=False)
        self.out_proj._is_residual = True  # type: ignore
        
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)
    
    def prune_params(self, zs_block):
        head_z = None; head_layer_z = None; hidden_z = None; qk_head_dim_z = None; vo_head_dim_z = None
        if "head_z" in zs_block:
            head_z = zs_block["head_z"].squeeze()
        
        if "head_layer_z" in zs_block:
            head_layer_z = zs_block["head_layer_z"].squeeze()
        
        if "hidden_z" in zs_block:
            hidden_z = zs_block["hidden_z"].squeeze()
        
        if "qk_head_dim_z" in zs_block:
            qk_head_dim_z = zs_block["qk_head_dim_z"].squeeze() # qk_head_dim is the same as hidden_z
            vo_head_dim_z = zs_block["vo_head_dim_z"].squeeze() # vo_head_dim is the same as hidden_z
            
            
        # update params #
        if head_z is not None:
            head_z_for_update = torch.repeat_interleave(head_z, self.head_dim)
            self.wv.weight.data = self.wv.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)
        if head_layer_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
        if hidden_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
        if qk_head_dim_z is not None:
            self.wq.weight.data = self.wq.weight.data.transpose(0, 1).mul(qk_head_dim_z).transpose(0, 1)
            self.wv.weight.data = self.wv.weight.data.transpose(0, 1).mul(vo_head_dim_z).transpose(0, 1)
        #################
        
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            print(f"    Head hidden: {len(hidden_z)} -> {len(remaining_index)}") 
            half = next(self.wq.parameters()).dtype == torch.float16
            self.wk = prune_linear_layer(self.wk, remaining_index, dim=1)
            self.wq= prune_linear_layer(self.wq, remaining_index, dim=1)
            self.wv = prune_linear_layer(self.wv, remaining_index, dim=1)
            self.out_proj = prune_linear_layer(self.out_proj, remaining_index)
            if half:
                self.wq.half()
                self.wk.half()
                self.wv.half()
                self.out_proj.half()
         
        to_prune_heads = turn_head_z(head_z, head_layer_z)
        len_to_prune_heads = len(to_prune_heads)
        if len_to_prune_heads == 0:
            print(f"    Heads: {self.n_heads} -> {self.n_heads}")
            return

        heads, index = find_pruneable_heads_and_indices(
            to_prune_heads, self.n_heads, self.head_dim, self.pruned_heads
        )
        
        qk_index = index; vo_index = index
        if qk_head_dim_z is not None:
            remaining_qk_index = torch.where(~qk_head_dim_z.eq(0))[0]
            remaining_vo_index = torch.where(~vo_head_dim_z.eq(0))[0]
            import numpy as np
            qk_index = torch.from_numpy(np.intersect1d(index.detach().cpu().numpy(), remaining_qk_index.detach().cpu().numpy())).to(index.device).to(index.dtype)
            vo_index = torch.from_numpy(np.intersect1d(index.detach().cpu().numpy(), remaining_vo_index.detach().cpu().numpy())).to(index.device).to(index.dtype)
            print(f"    QKVO dims: {len(hidden_z)} -> {len(qk_index)}")
        
        # Prune linear layers
        # setting layers to be None if all the heads are pruned
        if len(index) == 0:
            self.wq = None
            self.wk = None
            self.wv = None
            self.out_proj = None
        else:
            half = next(self.wq.parameters()).dtype == torch.float16
            self.wq = prune_linear_layer(self.wq, qk_index)
            self.wk = prune_linear_layer(self.wk, qk_index)
            self.wv = prune_linear_layer(self.wv, vo_index)
            self.out_proj = prune_linear_layer(self.out_proj, vo_index, dim=1)
            if half:
                self.wq.half()
                self.wk.half()
                self.wv.half()
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

        # qkv = self.Wqkv(x)
        # query, key, value = qkv.chunk(3, dim=2)
        if self.wq is None:
            return None, None, past_key_value

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        if qk_head_dim_z is not None:
            query = query.mul(qk_head_dim_z)
            value = value.mul(vo_head_dim_z)
        
        query_padding_mask = None
        if key_padding_mask is not None:
            query_padding_mask = key_padding_mask[:, -query.size(1):]
        
        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]

        # b, s, d = query.shape
        query = rearrange(query, 'b s (h d) -> b h s d', h=self.n_heads)
        key = rearrange(key, 'b s (h d) -> b h s d', h=self.n_heads)
        value = rearrange(value, 'b s (h d) -> b h s d', h=self.n_heads)
        
        kv_seq_len = key.size(2)
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=offset)

        offset = 0
        if past_key_value is not None:
            if len(past_key_value) != 0:
                offset = past_key_value[0].shape[-2]
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)
                past_key_value = (key, value)

        if self.attn_fn == flash_attn_fn:
            query = rearrange(query, 'b h s d -> b s h d')
            key = rearrange(key, 'b h s d -> b s h d')
            value = rearrange(value, 'b h s d -> b s h d')
            context, attn_weights = self.attn_fn(
                query,
                key,
                value,
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


class LlamaMLP(nn.Module):
    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.gate_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False, device=device)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False, device=device)
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False, device=device)
        self.down_proj._is_residule = True # type: ignore

    def prune_params(self, zs_block):
        intermediate_z = zs_block.get("intermediate_z", None)
        mlp_z = zs_block.get("mlp_z", None)
        hidden_z = zs_block.get("hidden_z", None)
        # update params #
        if intermediate_z is not None:
            self.up_proj.weight.data = self.up_proj.weight.data.transpose(0, 1).mul(intermediate_z.squeeze(0)).transpose(0, 1)
        if mlp_z is not None:
            self.down_proj.weight.data = self.down_proj.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
        if hidden_z is not None:
            self.down_proj.weight.data = self.down_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1) 
        #################

        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            print(f"    FFN hidden dim: {len(hidden_z)} -> {len(remaining_index)}")
            half = next(self.up_proj.parameters()).dtype
            self.up_proj = prune_linear_layer(self.up_proj, remaining_index, dim=1)
            self.gate_proj = prune_linear_layer(self.gate_proj, remaining_index, dim=1)
            self.down_proj = prune_linear_layer(self.down_proj, remaining_index, dim=0)
            if half == torch.float16:
                self.up_proj = self.up_proj.half()
                self.gate_proj = self.gate_proj.half()
                self.down_proj = self.down_proj.half()
            
        keep_dim = turn_mlp_z(intermediate_z, mlp_z)
        device = self.up_proj.weight.device
        if len(keep_dim) == self.up_proj.weight.shape[0]:
            print(f"    FFN intermediate dim: {self.cfg.intermediate_size} -> {len(keep_dim)}")
            return 
            
        if len(keep_dim) == 0:
            self.up_proj = None; self.down_proj = None; self.gate_proj = None
        else:
            keep_dim_index = torch.tensor(keep_dim).long().to(device)
            half = next(self.up_proj.parameters()).dtype
            self.up_proj = prune_linear_layer(self.up_proj, keep_dim_index, dim=0)
            self.gate_proj = prune_linear_layer(self.gate_proj, keep_dim_index, dim=0)
            self.down_proj = prune_linear_layer(self.down_proj, keep_dim_index, dim=1)
            if half == torch.float16:
                self.up_proj = self.up_proj.half()
                self.gate_proj = self.gate_proj.half()
                self.down_proj = self.down_proj.half()
        print(f"    FFN intermediate dim: {self.cfg.intermediate_size} -> {len(keep_dim)}")
        
    def forward(self, x, retain_grad=False, intermediate_z=None, mlp_z=None, hidden_z=None):
        if self.up_proj is None:
            return None
        gate = F.silu(self.gate_proj(x))
        up_v = self.up_proj(x)
        if retain_grad:
            self.up_v = up_v
            if self.up_v.requires_grad:
                self.up_v.retain_grad()
        if intermediate_z is not None:    
            up_v *= intermediate_z
        down_v = self.down_proj(gate * up_v)
        
        if retain_grad:
            self.output = down_v
            if self.output.requires_grad:
                self.output.retain_grad()
        
        if mlp_z is not None:
            down_v = down_v * mlp_z
            
        if hidden_z is not None:
            down_v = down_v * hidden_z
            
        return down_v
            

def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'{tensor.dtype=} must be in {valid_dtypes=}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors ({tensor.is_cuda=}).')

def normal_attn_fn(
    query,
    key, 
    value,
    attention_mask=None,
    head_z=None
):
    bsz, n_heads, q_len, head_dim = query.shape
    dim = n_heads * head_dim
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    # upcast attention to fp32
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value) # (bsz, n_heads, q_len, head_dim)
    if head_z is not None:
        attn_output *= head_z.unsqueeze(-1)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, dim)
    return attn_output
    
def flash_attn_fn(
    query,
    key,
    value,
    softmax_scale=None,
    attn_bias=None,
    query_padding_mask=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    head_z=None,
    
):
    try:
        from flash_attn import bert_padding  # type: ignore
        from flash_attn import flash_attn_interface  # type: ignore
    except ImportError as e:
        raise e

    # check_valid_inputs(query, key, value)

    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')

    batch_size, seqlen = query.shape[:2]

    if query_padding_mask is None:
        query_padding_mask = torch.ones((batch_size, seqlen), dtype=torch.bool, device=query.device)
    if key_padding_mask is None:
        key_padding_mask = torch.ones((batch_size, seqlen), dtype=torch.bool, device=key.device)

    query_unpad, indices_q, cu_seqlens_q, max_seqlen_q = bert_padding.unpad_input(
        query, query_padding_mask)
    # query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    key_unpad, _, cu_seqlens_k, max_seqlen_k = bert_padding.unpad_input(
        key, key_padding_mask)
    # key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    value_unpad, _, _, _ = bert_padding.unpad_input(value, key_padding_mask)
    # value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    dropout_p = dropout_p if training else 0.0
    
    output_unpad = flash_attn_interface.flash_attn_unpadded_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=is_causal,
        return_attn_probs=needs_weights)

    if head_z is not None:
        output_unpad = output_unpad * head_z # 1 * h * 1
    output = bert_padding.pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen)
    return output, None

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
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
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def prepare_decoder_attention_mask(input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(input_shape, inputs_embeds.dtype).to(inputs_embeds.device)

    return combined_attention_mask

