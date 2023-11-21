
import os
import time
from typing import Any, Dict, List

import torch
from composer import Callback, Logger, State
from composer.loggers import Logger
from composer.utils import dist
from torch.nn import functional as F


class DebugCallback(Callback):
    def batch_start(self, state: State, logger: Logger) -> None:
        for b in state.batch["input_ids"]:
            print(b) 
        
def scatter_add_loss(instance_loss, set_ids):
    unique_set, counts = set_ids.unique(return_counts=True, sorted=True)
    set_loss = torch.zeros(set_ids.max().item() + 1, dtype=instance_loss.dtype, device=instance_loss.device).scatter_add_(0, set_ids, instance_loss)
    set_loss = set_loss[set_loss > 0]
    set_loss = set_loss / counts.to(set_loss.dtype)
    return set_loss, unique_set

def get_set_loss(instance_loss, set_ids):
    unique_set, counts = set_ids.unique(return_counts=True, sorted=True)
    set_loss = []
    for set_id in unique_set:
        set_loss.append(instance_loss[set_ids == set_id].mean().item())
    return torch.tensor(set_loss, device=instance_loss.device), unique_set

class DoReMiCallback(Callback):
    def init(self, state: State, logger: Logger):
        self.n_domains = len(state.model.set_names)
        # per domain average diff from 7b
        state.model.per_domain_avg_diff = [0.34828198, 0.38631809, 0.34036183, 0.39682519, 0.58600819, 0.32024884, 0.38567674] # fixed, TODO
        state.model.current_domain_weight = [1 / self.n_domains] * self.n_domains 
        print("Initialize parameters for doremi")
        
    def before_train_batch(self, state: State, logger: Logger):
        device_batch = state.batch
        with torch.no_grad():
            output = state.model.forward(device_batch)
        ref_output = state.model.ref_forward(device_batch)
        logits = output["logits"]
        ref_logits = ref_output["logits"]
        
        targets = state.model.get_targets(device_batch)
        batch_size, seq_len = device_batch["input_ids"].shape
        
        instance_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                        targets.view(-1),
                                        ignore_index=-100,
                                        reduction='none').reshape(batch_size, seq_len).mean(dim=-1)
        ref_instance_loss = F.cross_entropy(ref_logits.view(-1, ref_logits.size(-1)),
                                    targets.view(-1),
                                    ignore_index=-100,
                                    reduction='none').reshape(batch_size, seq_len).mean(dim=-1)
        set_ids = device_batch["set"]

        all_instance_loss = torch.cat(dist.all_gather(instance_loss))
        all_ref_instance_loss = torch.cat(dist.all_gather(ref_instance_loss))
        all_set_ids = torch.cat(dist.all_gather(set_ids))
        dist.barrier() 

        set_loss, unique_set = get_set_loss(all_instance_loss.float(), all_set_ids)
        ref_set_loss, _ = get_set_loss(all_ref_instance_loss.float(), all_set_ids)
        dist.barrier()

        excess_loss = set_loss - ref_set_loss

        print(excess_loss)
        eta = 1
        c = 1e-4
        
        current_domain_weight = state.model.current_domain_weight

        current_domain_ids = unique_set.cpu().tolist()
        for i in range(self.n_domains):
            if i in current_domain_ids:
                per_domain_score = max(excess_loss[current_domain_ids.index(i)].item(), 0)
                state.model.per_domain_avg_diff[i] = per_domain_score
        updated_alpha = torch.log(torch.tensor(current_domain_weight)) + eta * torch.tensor(state.model.per_domain_avg_diff)
        updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
        updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains
        state.model.current_domain_weight = updated_domain_weights.tolist()

class ImpCallback(Callback):
    """ Callback for calculating importance scores (not used) """
    def __init__(
        self,
        device_train_microbatch_size=8,
        total_exs=1,        
        save_folder: str = None,
    ):
        self.batch_num = 0
        self.t = time.time()
        self.device_train_microbatch_size = device_train_microbatch_size
        self.total_exs = total_exs
        self.save_folder = save_folder
        
    def plug_in_pruned_steps(self, state: State, logger: Logger):
        input_ids = state.batch["input_ids"]
        state.batch["retain_grad"] = torch.BoolTensor([True] * len(input_ids)).to(input_ids.device)
            
    def batch_start(self, state: State, logger: Logger):
        self.plug_in_pruned_steps(state, logger)
        
    def after_backward(self, state: State, logger: Logger) -> None:
        def attach_module(module, name):
            output = getattr(module, name)
            grad = output.grad
            imp = (output * grad).view(-1, grad.shape[-1]).sum(0).detach()
            
            attach_name = name + "_imp"
            accumulated_imp = getattr(module, attach_name, None)
            
            if accumulated_imp is None:
                setattr(module, attach_name, imp)
            else:
                updated_imp = accumulated_imp + imp
                setattr(module, attach_name, updated_imp)
                
            state.model.zero_grad()
            state.optimizers[0].zero_grad()
                
        for layer in state.model.model.transformer.blocks:
            attach_module(layer.attn, "context") # (b * seq) * dim 
            attach_module(layer.attn, "output") # b * seq * dim
            attach_module(layer.mlp, "up_v") # b * seq * int_dim
            attach_module(layer.mlp, "output") # b * seq * dim
        
        self.batch_num += 1
        self.t = time.time() - self.t
        passed_exs = self.batch_num * self.device_train_microbatch_size * dist.get_world_size()
        print(f"[{passed_exs}/{self.total_exs}]", "elapsed and took", round(self.t, 2), "seconds.")
        if passed_exs % 10240 == 0:
            self.save_imp(state, passed_exs)
    
    def save_imp(self, state, passed_exs):
        d = []
        for layer in state.model.model.transformer.blocks:
            layer_d = {"attn": layer.attn.output_imp, "mlp_int": layer.mlp.up_v_imp, "mlp": layer.mlp.output_imp, "context": layer.attn.context_imp}
            d.append(layer_d)
        torch.save(d, os.path.join(self.save_folder.replace("{run_name}", state.run_name), f"imp-{passed_exs}.pt"))

    def after_train_batch(self, state: State, logger: Logger) -> None:
        import sys; sys.exit()
