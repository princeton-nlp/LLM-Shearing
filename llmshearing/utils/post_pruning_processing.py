import glob
import os

import torch

from llmshearing.models.composer_llama import ComposerMosaicLlama
from llmshearing.utils.utils import load_weights


def prune_and_save_model(path):
    """ prune and save the model after pruning """
    outpath = os.path.dirname(path) + f"/pruned-{os.path.basename(path)}"
    config_file = os.path.join(os.path.dirname(path), "config.pt")
    assert os.path.exists(config_file), f"Config file {config_file} does not exist"
    
    cfg = torch.load(config_file).model
    if cfg.l0_module.target_model is not None:
        cfg.l0_module.eval_target_model = True # hack
   
    model = ComposerMosaicLlama(cfg)
    weights = load_weights(path)
    
    ree = model.load_state_dict(weights, strict=False)
    print(ree)
    
    model.prune_params() 
    model.model.l0_module = None
    model_state_dict = model.state_dict()
    new_weights = change_keys(model_state_dict)
    torch.save(new_weights, outpath)
    print("Saved pruned model to path: ", outpath)
    

def change_keys(weights, output_file=None):
    """ rename the keys in the weight file to match the new model """
    exitsing_layers = []
    for key in weights:
        if "blocks" in key and "rotary" not in key:
            layer = int(key[key.index("blocks") + len("blocks."):].split(".")[0])
            if layer not in exitsing_layers:
                exitsing_layers.append(layer)
    exitsing_layers = sorted(exitsing_layers)
    print("Existing layers: ", len(exitsing_layers), exitsing_layers)
    
    new_weights = {}
    for key in weights:
        if "rotary" in key:
            continue
        if "blocks" in key:
            layer_index = key.index("blocks") + len("blocks.")
            text_before_layer_index = key[:layer_index]
            layer = int(key[layer_index:].split(".")[0])
            text_after_layer_index = key[layer_index + len(str(layer)) + 1:]
            current_layer = exitsing_layers.index(layer)
            new_key = text_before_layer_index + str(current_layer) + "." + text_after_layer_index
            print("Old param key:", key)
            print("New param key:", new_key)
        else:
            new_key = key
        new_weights[new_key] = weights[key]
    if output_file is not None:
        torch.save(new_weights, output_file)
    else:
        return new_weights
    
if __name__ == "__main__":
    import fire
    fire.Fire()