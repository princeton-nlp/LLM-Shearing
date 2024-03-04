# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings
from types import MethodType
from typing import Any, Dict

import torch
from composer import Logger, State, Trainer
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.core import Evaluator, Event
from composer.loggers import FileLogger
from composer.optim import DecoupledAdamW
from composer.utils import dist, get_device, reproducibility
from llmfoundry.optim import (DecoupledAdaLRLion, DecoupledClipLion,
                              DecoupledLionW, DecoupledLionW_8bit)
from llmfoundry.utils.builders import (build_algorithm, build_callback,
                                       build_logger, build_scheduler)
from llmfoundry.utils.config_utils import (log_config, pop_config,
                                           update_batch_size_info)
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch import nn
from torch.optim.optimizer import Optimizer

from llmshearing.callbacks.callbacks import DebugCallback
from llmshearing.callbacks.dynamic_loading_callback import \
    DynamicLoadingCallback
from llmshearing.callbacks.pruning_callback import PruningCallback
from llmshearing.datasets.load_text_dataloader import build_text_dataloader
from llmshearing.models.model_registry import COMPOSER_MODEL_REGISTRY

streaming.base.util.clean_stale_shared_memory()

def is_one_hour(run_name: str):
    """ Check if the run name is for one hour training. """
    return run_name.startswith("ONE_HOUR")

def exit_batch_checkpoint(self, state: State, logger: Logger):
    """ Exit the program after saving the checkpoint. """
    if self.save_interval(state, Event.BATCH_CHECKPOINT) and self.last_checkpoint_batch != state.timestamp.batch:
        self._save_checkpoint(
            state,
            logger,
        )
        print("Ending program at batch", state.timestamp.batch)
        print(self.folder)
        sys.exit()
        
def validate_config(cfg: DictConfig):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if 'eval_loader' in cfg:
        loaders.append(cfg.eval_loader)
    
def build_composer_model(cfg: DictConfig):
    """ build the composer model """
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    return COMPOSER_MODEL_REGISTRY[cfg.name](cfg)

def load_weights(cfg: DictConfig):
    """ load weights """
    if cfg.model.get('path', None):
        state_dict = torch.load(cfg.model.path) # for loading pre-trained llama
        if "state" in state_dict:
            state_dict = state_dict["state"]["model"] 
        print("Loaded model from path: ", cfg.model.path)
        return state_dict
    return None

def load_state_dict(model: nn.Module, state_dict: Dict[str, Any]):
    """ load state dict to the model """
    result = model.load_state_dict(state_dict, strict=False)
    print("Model load state dict result: ", result)
    print("Having missing rotary_emb.inv_freq keys is normal")

def build_optimizer(model: torch.nn.Module, name: str,
                    optimizer_config: Dict[str, Any]) -> Optimizer:
    """ 
        build optimizer that consists of three groups of parameters:
        - main_model_params: parameters of the main model
        - l0_module_params: parameters of the l0 module
        - lagrange_params: parameters of the lagrange multipliers
    """    
    param_groups = {}
    main_model_params = [p for n, p in model.named_parameters() if "l0_module" not in n]
    l0_module_params = [p for n, p in model.named_parameters() if "l0_module" in n and "lambda" not in n]
    lagrange_params = [p for n, p in model.named_parameters() if "l0_module" in n and "lambda" in n]

    param_groups = [{"params": main_model_params, "lr": optimizer_config.lr}]
    lag_lr = pop_config(optimizer_config, "lag_lr")
    if len(l0_module_params) > 0:
        param_groups.extend([{"params": l0_module_params, "lr": lag_lr}, {"params": lagrange_params, "lr": -(lag_lr)}])
    
    for i, group in enumerate(param_groups):
        print(f"Group {i}:", f"{len(group['params'])} tensors", f"{sum(p.numel() for p in group['params'])} params", f"{group['lr']:.2e} lr")
            
    if name == 'decoupled_adamw':
        return DecoupledAdamW(param_groups, **optimizer_config)
    elif name == 'decoupled_lionw':
        return DecoupledLionW(param_groups, **optimizer_config)
    elif name == 'clip_lion':
        return DecoupledClipLion(param_groups, **optimizer_config)
    elif name == 'adalr_lion':
        return DecoupledAdaLRLion(param_groups, **optimizer_config)
    elif name == 'decoupled_lionw_8b':
        return DecoupledLionW_8bit(param_groups, **optimizer_config)
    else:
        raise ValueError(f'Not sure how to build optimizer: {name}')
    
def main(cfg):
    """ Main training function """
    print("Start running ")
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=f'torch.distributed.*_base is a private function and will be deprecated.*'
    )
    cfg.dist_timeout = cfg.get('dist_timeout', 1800.0)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)
    
    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)
    
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message='torch.distributed.*_base is a private function and will be deprecated.*'
    )

    reproducibility.seed_all(cfg.seed)

    # Run Name
    if cfg.get('run_name') is None:
        cfg.run_name = os.environ.get('COMPOSER_RUN_NAME', 'llm')

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None
    
    # Restrict model init_device to 'meta' and 'cpu',
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_device = cfg.model.get('init_device', 'cpu')
    assert init_device in ['meta', 'cpu']
    if fsdp_config is None and init_device == 'meta':
        warnings.warn(
            "Using `cfg.model.init_device='meta'` is only valid when using FSDP! " +\
            "Reverting to `cfg.model.init_device='cpu'`.")
        cfg.model.init_device = 'cpu'

     # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get('loggers') or {}).items()
    ]
    
    save_folder = cfg.save_folder.replace('{run_name}', cfg.run_name)
    filename = f"{save_folder}/logs.txt"
    count = 1
    
    while os.path.exists(filename):
        print(f"File {filename} already exists")
        filename  = f"{save_folder}/logs_{count}.txt"
        count += 1
    print(f"Logging to {filename}")
    loggers.append(FileLogger(filename=filename,
                             buffer_size=1,
                             flush_interval=50))
    
    # Build Model
    print('Initializing model...')
    if cfg.callbacks.data_loading.dynamic:
        cfg.model.set_names = cfg.callbacks.data_loading.set_names
    model = build_composer_model(cfg.model)
    print(model)
    print(cfg.model.l0_module)

    state_dict = load_weights(cfg)
    if state_dict is not None:
        load_state_dict(model, state_dict)
     
    cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f'{cfg.n_params=:.2e}')
    if hasattr(model, 'num_fwd_flops'):
        print(f'{model.num_fwd_flops=:.2e}')
    
    # set names has to be part of the config    
    assert getattr(cfg.callbacks.data_loading, 'set_names', None) is not None, "please specify the set (domain) names in the config"
    
    # Dataloaders
    print('Building train loader...')
    train_loader = build_text_dataloader(cfg.train_loader,
                                         cfg.device_train_batch_size,
                                         cfg.callbacks.data_loading.dynamic,
                                         cfg.callbacks.data_loading.set_names,
                                         proportion=cfg.callbacks.data_loading.proportion)
    print('Building eval loader...')
    evaluators = []
    if 'eval_loader' in cfg:
        # eval data is never loaded dynamically
        eval_loader = Evaluator(label='eval',
                                dataloader=build_text_dataloader(
                                cfg.eval_loader,
                                cfg.device_eval_batch_size,
                                dynamic=False,
                                set_names=cfg.callbacks.data_loading.set_names,                                proportion=None),
                                metric_names=list(model.train_metrics.keys()))
        evaluators.append(eval_loader)

    # Optimizer
    optimizer = build_optimizer(model, cfg.optimizer.pop("name"), cfg.optimizer)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler.pop("name"), cfg.scheduler)

    # Callbacks
    callbacks = []
    data_loading_config = pop_config(cfg.callbacks, 'data_loading')
    if data_loading_config.dynamic:
        dl_callback = DynamicLoadingCallback(target_loss=data_loading_config.target_loss,
                                             proportion=data_loading_config.proportion,
                                             set_names=data_loading_config.set_names,
                                             update_type=data_loading_config.update_type)
        callbacks.append(dl_callback)
    callbacks += [
        build_callback(name, callback_cfg)
        for name, callback_cfg in (cfg.get('callbacks') or {}).items()
    ]
    if model.model.l0_module is not None: # pruning callback
        callbacks.append(PruningCallback(save_folder=cfg.save_folder))
    
    # callbacks.append(DebugCallback())
    
    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.get('algorithms') or {}).items()
    ]

    # Build the Trainer
    print('Building trainer...')
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        console_log_interval=cfg.get('console_log_interval', '1ba'),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=cfg.get('device_train_microbatch_size', 'auto'),
        fsdp_config=fsdp_config,  # type: ignore
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep', -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
        load_ignore_keys=cfg.get('load_ignore_keys', None),
        python_log_level=cfg.get('python_log_level', None),
        dist_timeout=cfg.dist_timeout,
        autoresume=cfg.autoresume,
    )
    
    # a setup for one hour training
    if is_one_hour(cfg.run_name):
        for callback in trainer.state.callbacks:
            if isinstance(callback, CheckpointSaver):
                callback.batch_checkpoint = MethodType(exit_batch_checkpoint, callback)
    
    if data_loading_config.dynamic:
        # reload the function that allows saving the used domain ids
        from llmshearing.datasets.state import _dataset_state_dict
        trainer.state._dataset_state_dict = MethodType(_dataset_state_dict, trainer.state)
        
    print('Logging config...')
    log_config(cfg)

    if cfg.get('eval_first', False):
        trainer.eval()

    print('Starting training...')
    trainer.fit()

    print('Done.')

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
     
    # save the config files 
    save_dir = cfg.save_folder.replace("{run_name}", cfg.run_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(cfg, save_dir + "/config.pt") 
    
    main(cfg)
    
