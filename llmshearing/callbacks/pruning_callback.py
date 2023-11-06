import os
import time
from typing import Any, Dict, List

import torch
from composer import Callback, Logger, State
from composer.loggers import Logger
from composer.utils import dist
from torch.nn import functional as F


class PruningCallback(Callback):
    """
        The interplay of pruning and the main training process is implemented fully based on the callback mechanism.
    """
    def __init__(self, save_folder: str = None) -> None:
        self.save_folder = save_folder

    def plug_in_pruned_steps(self, state: State, logger: Logger):
        """ Hack: Add pruned_steps to the batch to calculate target sparsity during the pruning warmup stage """
        if getattr(state.model.model, "l0_module", None) is not None:
            input_ids = state.batch["input_ids"]
            state.batch["pruned_steps"] = torch.LongTensor([state.timestamp.batch.value] * len(input_ids)).to(input_ids.device)
            
    def batch_start(self, state: State, logger: Logger):
        self.plug_in_pruned_steps(state, logger)
    
    def eval_batch_start(self, state: State, logger: Logger):
        self.plug_in_pruned_steps(state, logger)
             
    def after_train_batch(self, state: State, logger: Logger) -> None:
        """ Log information from the L0 module after each training batch """
        l0_output = state.outputs["l0_output"]
        if l0_output is not None:
            logger.log_metrics({f'metrics/train/{name}': val.cpu().item() if torch.is_tensor(val) else val for (name, val) in l0_output[1].items()})
    
    def eval_end(self, state: State, logger: Logger) -> None:
        """ Save the deterministic masks after each evaluation for analysis """
        zs = state.outputs["zs"]
        zs = {key: zs[key].detach().float().cpu().numpy() for key in zs}
        step = state.timestamp.batch.value
        torch.save(zs, os.path.join(self.save_folder.replace("{run_name}", state.run_name), f"zs_s{step}.pt"))
