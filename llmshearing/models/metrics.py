
from typing import Mapping, Optional, Union

import torch
from composer.metrics.nlp import LanguageCrossEntropy
from torch import Tensor
from torchmetrics import Metric


class DomainLanguageCrossEntropy(LanguageCrossEntropy):
    """ This class is used to calculate the cross entropy loss for each domain. """
    def __init__(self, dist_sync_on_step: bool = False, ignore_index: int = -100, set_name="arxiv"):
        super().__init__(dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index)
        self.set_name = set_name

    def update(self, output: Union[Mapping, Tensor], target: Tensor) -> None:
        if isinstance(output, Mapping):
            logits = output['logits']
        elif isinstance(output, Tensor):
            logits = output
        else:
            raise Exception(f'Type {type(output)} for the output is unsupported.')

        target = target.view(-1)
        logits = logits.view(target.shape[0], -1)
        losses = self.loss_fn(logits, target)

        total_items = (target != self.ignore_index).sum()
        self.total_items += total_items  #type: ignore (third-party)

        # accumulate loss over all batches
        self.sum_loss += losses.to(torch.float32)


class DomainCount(Metric):
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False, set_name="arxiv", set_index=0):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_cpu=False)

        self.add_state('idx', default=[], dist_reduce_fx=None)        
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.set_name = set_name 
        self.set_index = set_index 

    def reset(self) -> None:
        """This method automatically resets the metric state variables to their default value."""
        # reset internal states
        self._cache = None
        self._is_synced = False
        
    def update(self, selected_sets: Union[Mapping, Tensor], idxs: Union[None, Tensor] = None) -> None:
        mask = selected_sets == self.set_index
        self.count += mask.sum()
        if idxs is not None:
            added_idxs = idxs[mask].contiguous()
            if added_idxs.numel() > 0: # would cause error when gather across workers 
                self.idx.append(added_idxs)
      
    def compute(self):
        return self.count 

class DomainWeight(Metric):
    full_state_update = False
    def __init__(self, domain_weight, set_name="arxiv", set_index=0):
        super().__init__(dist_sync_on_step=False)
        self.domain_weight = domain_weight
        self.set_index = set_index

    def update(self, domain_weight) -> None:
        self.domain_weight = domain_weight
      
    def compute(self):
        return self.domain_weight 
    
class DomainDiff(Metric):
    full_state_update = False
    def __init__(self, domain_diff, set_name="arxiv", set_index=0):
        super().__init__(dist_sync_on_step=False)
        self.domain_diff = domain_diff
        self.set_index = set_index

    def update(self, domain_diff) -> None:
        self.domain_diff = domain_diff
      
    def compute(self):
        return self.domain_diff 
    