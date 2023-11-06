import glob
import os

import torch

def load_weights(path):
    """ load model weights from a path """
    if not isinstance(path, str):
        path = str(path)
    ckpt_paths = [path]
    if not path.endswith(".pt"):
        ckpt_paths = [path + "/latest-rank0.pt"]
        if not os.path.exists(ckpt_paths[0]):
            ckpt_paths = glob.glob(path + "/pytorch_model*bin")
    
    state_dict = {}
    for p in ckpt_paths: 
        if torch.cuda.is_available():
            p_weight = torch.load(p)
        else:
            p_weight = torch.load(path, map_location=torch.device('cpu'))
        if "state" in p_weight:
            state_dict.update(p_weight["state"]["model"])
        else:
            state_dict.update(p_weight)
    print("Loaded model from path: ", path)
    return state_dict
