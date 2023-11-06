import torch
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmshearing.models.composer_llama import ComposerMosaicLlama


def construct_example_cfg(model_size, path=None, add_l0_module=False):
    """ construct example cfg for mosaicml llama models """
    if model_size == "toy":
        cfg = om.create({"name": "mosaic_llama_100m", "init_device": "cpu", "d_model": 768, 
                         "n_heads": 12, "n_layers": 12, "intermediate_size": 3072})
    if model_size == "7B":
        cfg = om.create({"name": "mosaic_llama_7b", "path": path, "init_device": "cpu",
                         "d_model": 4096, "n_heads": 32, "n_layers": 32, "intermediate_size": 11008})
    if model_size == "13B":
        cfg = om.create({"name": "mosaic_llama_13b", "path": path,"init_device": "cpu", "d_model": 5120, 
                         "n_heads": 40, "n_layers": 40, "intermediate_size": 13824})
    if model_size == "30B":
        cfg = om.create({"name": "mosaic_llama_30b", "path": path,"init_device": "cpu", "d_model": 6656, "n_heads": 52, "n_layers": 60, "intermediate_size": 17920})
    if model_size == "65B":
        cfg = om.create({"name": "mosaic_llama_65b", "path": path,"init_device": "cpu", "d_model": 8192, "n_heads": 64, "n_layers": 80, "intermediate_size": 22016})
    
    # add default values
    cfg = om.merge(cfg, om.create({"max_seq_len": 4096, "vocab_size": 32000, "init_std": 0.02, "attn_pdrop": 0.0, "resid_pdrop": 0.0, "emb_pdrop": 0.0, "attn_impl": "flash", "rms_norm_eps": 1e-5}))
    if add_l0_module:
        cfg["l0_module"] = {"start_sparsity": 0, "target_sparsity": 0.6, "pruning_modules": ["head", "head_layer", "mlp", "intermediate", "hidden"], "lagrangian_warmup_steps": "320ba"}
    return cfg

def test_two_matrix(a, b, desc=""):
    """ test if two matrix are equal """
    s1 = a.sum().item(); s2 = b.sum().item() if b is not None else torch.tensor(0).to(a.device).to(a.dtype)
    try:
        assert abs(s1 - s2) < 1e-3
    except:
        print(f"[{desc}] failed! sums are not equal: {s1} vs {s2}")
        return 
    print(f"[{desc}] passed! sums are equal: {s1} vs {s2}")


if __name__ == "__main__":
    import sys
    
    hf_llama2_path = sys.argv[1]
    composer_llama2_path = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(hf_llama2_path)
    text = "Chamath Palihapitiya (born 3 September 1976)[1] is a Sri Lankan-born Canadian and American venture capitalist, engineer, SPAC sponsor, founder and CEO of Social Capital. Palihapitiya was an early senior executive at Facebook, working at the company from 2007 to 2011. Following his departure from Facebook, Palihapitiya started his fund, The Social+Capital Partnership, through which he invested in several companies, including Yammer and Slack. "
    input_ids = tokenizer.encode(text, return_tensors="pt")


    # check if they have the same naming convention
    hf_model = AutoModelForCausalLM.from_pretrained(hf_llama2_path)
    hf_loss = hf_model(input_ids, labels=input_ids).loss

    cfg = construct_example_cfg("7B")
    composer_model = ComposerMosaicLlama(cfg)
    # rotary_emb.inv_freq can be missing
    composer_model.load_state_dict(torch.load(composer_llama2_path), strict=False)

    input_ids = input_ids.cuda()
    composer_model.bfloat16().cuda()
    hf_model.bfloat16().cuda()

    logits1 = hf_model(input_ids, labels=input_ids).logits.mean()
    logits2 = composer_model({"input_ids": input_ids})["logits"].mean()

    test_two_matrix(logits1, logits2, "HF vs. Composer")

