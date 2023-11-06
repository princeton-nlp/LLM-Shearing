from typing import Any, Dict

from llmshearing.callbacks.dynamic_loading_callback import \
    DynamicLoadingCallback


def _dataset_state_dict(self) -> Dict[str, Any]:
    """Collect the state dict(s) of our train and eval dataset(s).

    Returns:
        Dict[str, Any]: The state dict(s).
    """
    obj = {
        'train': None,
        'eval': {},
    }

    dataset = self._dataset_of(self.train_dataloader)
    domains = dataset.set_names
    for callback in self.callbacks:
        if isinstance(callback, DynamicLoadingCallback):
            break
    
    if hasattr(dataset, 'state_dict'):
        num_samples = int(self.timestamp.sample_in_epoch.value)
        trained_num = 0
        for i, domain in enumerate(domains):
            trained_num += len(callback.used_domain_ids[i])
        assert num_samples == trained_num
        obj['train'] = dataset.state_dict(callback.used_domain_ids, True)  # pyright: ignore

    for evaluator in self.evaluators:
        dataset = self._dataset_of(evaluator)
        if hasattr(dataset, 'state_dict'):
            # Don't save eval sample because we do not checkpoint during eval.
            obj['eval'][evaluator.label] = dataset.state_dict(0, True)  # pyright: ignore
    
    return obj