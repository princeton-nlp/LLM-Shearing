""" test llama pruning """
from copy import deepcopy

import torch
from examples.llm.src.models.l0_module import L0Module
from examples.llm.src.models.mosaic_llama_v2 import (ComposerMosaicLlama,
                                                     LlamaRMSNorm)


def get_l0_module(config):
    l0_module = L0Module(config, lagrangian_warmup=200, target_sparsity=0.5)
    return l0_module    

def get_full_zs(l0_module, ones=False, half=False):
    with torch.no_grad():
        zs = l0_module()
        for key in zs:
            if ones:
                zs[key].fill_(1.)
            else:
                zs[key] = torch.FloatTensor(zs[key].shape).uniform_().abs().to(zs[key].device)
    if half:
        for key in zs:
            zs[key] = zs[key].half()
    return zs

def zero_out_zs(z, percentage):
    mask = torch.FloatTensor(z.shape).uniform_().abs() > percentage
    mask = mask.to(z.device)
    z = z * mask
    return z

def zero_out_qk_vo_head_dims(z, percentage, num_heads=12):
    # layer * dim
    dim = z.shape[-1] // num_heads
    zero_num = int(dim * percentage)
    
    reshaped_z = z.reshape(-1, dim)
    for i in range(reshaped_z.shape[0]):
        reshaped_z[i][torch.randperm(dim)[:zero_num]] = 0
    z = reshaped_z.reshape(z.shape)
    return z

def zero_out_all_zs(zs, percentage, num_heads=12):
    for key in zs:
        if key in percentage:
            if "qk" in key or "vo" in key:
                zs[key] = zero_out_qk_vo_head_dims(zs[key], percentage[key], num_heads=num_heads)
            else:
                zs[key] = zero_out_zs(zs[key], percentage[key])
    return zs

def build_composer_model(cfg):
    model = ComposerMosaicLlama(cfg)
    if cfg.get('path', None):
        path = cfg.path
        state_dict = torch.load(path)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded model from path: ", cfg.path)
    return model

def load_input_ids(cuda=True):
    input_ids = torch.tensor([[1, 910, 338, 263, 2107, 2462, 29991]])
    if cuda:
        input_ids = input_ids.cuda()
    return input_ids

def forward(model, input_ids, zs):
    batch = {"input_ids": input_ids, "labels": input_ids}
    batch.update(zs)
    outputs = model(batch)
    loss = model.loss(outputs, batch)["ce_loss"]
    return loss

# passed 
def test_full_z(model, l0_module, half=False, ones=False):
    """
    Compare the loss of 
        - original model forward
        - model forward with full zs
    """
    print(test_full_z.__doc__)
    input_ids = load_input_ids()
    
    zs = get_full_zs(l0_module, half=half, ones=ones)
    model1 = deepcopy(model).cuda()
    if half:
        model1 = model1.half()
    model1.prune_params(zs)
    loss1 = forward(model1, input_ids, zs={}) 

    model2 = deepcopy(model).cuda()
    if half:
        model2 = model2.half()
    loss2 = forward(model2, input_ids, zs)
    print(f"loss1: {loss1.item()}, loss2: {loss2.item()}")
    
    if loss1.item() != loss2.item():
        print("test_full_z failed!")
    else:
        print("test_full_z passed!")

# passed
def test_Shearing_LayerNorm(l0_module):
    from copy import deepcopy

    zs = get_full_zs(l0_module, half=True)
    zs["hidden_z"] = zero_out_zs(zs["hidden_z"], 0.3)
    remaining_index = zs["hidden_z"].squeeze().nonzero().squeeze()
    
    hidden_dim = len(zs["hidden_z"]) 
    layernorm1 = LlamaRMSNorm(hidden_dim).cuda()
    layernorm2 = deepcopy(layernorm1)

    input = torch.randn(2, 3, hidden_dim).cuda()
    out1 = layernorm1(input, zs["hidden_z"])
    out1 = torch.index_select(out1, dim=-1, index=remaining_index)
    
    # layernorm2.weight = torch.nn.Parameter(layernorm2.weight.mul(zs["hidden_z"].squeeze())[remaining_index])
    layernorm2.prune_params(zs["hidden_z"])
    compressed_input = torch.index_select(input, dim=-1, index=remaining_index)
    out2 = layernorm2(compressed_input)
    assert out1.sum().item() == out2.sum().item()
    print("test_Shearing_LayerNorm passed!")

def nice_print(v1, v2):
    if torch.is_tensor(v1): v1 = v1.detach().cpu().numpy().item()
    if torch.is_tensor(v2): v2 = v2.detach().cpu().numpy().item()
    print("v1:", v1)
    print("v2:", v2)

def eval(v1, v2, case_num=0):
    nice_print(v1, v2)
    if torch.isclose(v1, v2):
        print(f"case {case_num} passed!")
    else:
        print(f"case {case_num} failed!")
             
# passed 
def test_Shearing_Attention(model, l0_module, half=False, ones=False):
    zs = get_full_zs(l0_module, half=True, ones=ones)
    device = next(model.parameters()).device
    
    attn = deepcopy(model.model.transformer.blocks[0].attn)
    hidden_states = torch.randn(2, 3, model.model.cfg.d_model).to(device)
    if half: hidden_states = hidden_states.half()
    
    def copy_module():
        attn1 = deepcopy(attn).cuda()
        attn2 = deepcopy(attn).cuda()
        if half:
            attn1 = attn1.half()
            attn2 = attn2.half()
        attn1.eval(); attn2.eval()
        return attn1, attn2
        
    # case 1
    print("\n[Testing Attention] case 1: All heads are pruned")
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0.3, "head_layer_z": 1.})
    head_z = corrected_zs["head_z"][0]; head_layer_z = corrected_zs["head_layer_z"][0]
    zs_block = {"head_z": head_z, "head_layer_z": head_layer_z}
    
    attn1, attn2 = copy_module(); attn1.prune_params(zs_block)
    
    with torch.no_grad():
        attn_output1, _, _ = attn1(hidden_states)
        attn_output2, _, _ = attn2(hidden_states, **zs_block)
        if attn_output1 is None and attn_output2.sum().item() == .0:
            print("case 1 passed!")
        else:
            v1 = attn_output1.sum(); v2 = attn_output2.sum()
            nice_print(v1, v2)
            print("case 1 failed!")

    # case 2
    print("\n[Testing Attention] case 2: A non-zero number of heads are pruned")
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0.3, "head_layer_z": 0.})
    head_z = corrected_zs["head_z"][0]; head_layer_z = corrected_zs["head_layer_z"][0]
    zs_block = {"head_z": head_z, "head_layer_z": head_layer_z}
    attn1, attn2 = copy_module(); attn1.prune_params(zs_block)
    
    with torch.no_grad():
        attn_output1, _, _ = attn1(hidden_states)
        attn_output2, _, _ = attn2(hidden_states, **zs_block)
        v1 = attn_output1.sum(); v2 = attn_output2.sum()
        eval(v1, v2, 2)
            
    # case 3
    print("\n[Testing Attention] case 3: No heads are pruned")
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0., "head_layer_z": 0.})
    head_z = corrected_zs["head_z"][0]; head_layer_z = corrected_zs["head_layer_z"][0]
    zs_block = {"head_z": head_z, "head_layer_z": head_layer_z}
    attn1, attn2 = copy_module(); attn1.prune_params(zs_block)
    
    with torch.no_grad():
        attn_output1, _, _ = attn1(hidden_states)
        attn_output2, _, _ = attn2(hidden_states, head_z=head_z, head_layer_z=head_layer_z)
        v1 = attn_output1.sum(); v2 = attn_output2.sum()
        eval(v1, v2, 3) 
    
    # case 4
    print("\n[Testing Attention] case 4: A non-zero number of heads are pruned and hidden dimensions are pruned")
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0.3, "head_layer_z": 0., "hidden_z": 0.3})
    head_z = corrected_zs["head_z"][0]; head_layer_z = corrected_zs["head_layer_z"][0]; hidden_z = corrected_zs["hidden_z"]
    zs_block = {"head_z": head_z, "head_layer_z": head_layer_z, "hidden_z": hidden_z}
    attn1, attn2 = copy_module(); attn1.prune_params(zs_block)
    
    input = hidden_states.mul(hidden_z)
    
    with torch.no_grad():
        remaining_dim = torch.where(~hidden_z.eq(0))[0]
        compressed_hidden_states = input[..., remaining_dim]
        attn_output1, _, _ = attn1(compressed_hidden_states)
        attn_output2, _, _ = attn2(input, head_z=head_z, head_layer_z=head_layer_z, hidden_z=hidden_z)
        v1 = attn_output1.sum(); v2 = attn_output2.sum()
        eval(v1, v2, 4)
    
    # case 5
    print("\n[Testing Attention] case 5: hidden dims are pruned")
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"hidden_z": 0.3})
    hidden_z = corrected_zs["hidden_z"]
    zs_block = {"hidden_z": hidden_z, "head_z": corrected_zs["head_z"][0]}
    attn1, attn2 = copy_module(); attn1.prune_params(zs_block)
    
    input = hidden_states.mul(hidden_z)
    
    with torch.no_grad():
        remaining_dim = torch.where(~hidden_z.eq(0))[0]
        compressed_hidden_states = input[..., remaining_dim]
        attn_output1, _, _ = attn1(compressed_hidden_states)
        attn_output2, _, _ = attn2(input, **zs_block)
        v1 = attn_output1.sum(); v2 = attn_output2.sum()
        eval(v1, v2, 5)
         

def test_Shearing_MLP(model, l0_module, half=False, ones=False):
    zs = get_full_zs(l0_module, half=half, ones=ones)
    mlp = deepcopy(model.model.transformer.blocks[0].mlp)
    device = next(model.parameters()).device
    
    def copy_module():
        mlp1 = deepcopy(mlp).cuda()
        mlp2 = deepcopy(mlp).cuda()
        if half:
            mlp1 = mlp1.half()
            mlp2 = mlp2.half()
        mlp1.eval(); mlp2.eval()
        return mlp1, mlp2
     
    def run(percentage):
        hidden_states = torch.randn(2, 3, model.model.cfg.d_model).to(device);
        if half: hidden_states = hidden_states.half()
        corrected_zs = zero_out_all_zs(deepcopy(zs), percentage)
        intermediate_z = corrected_zs["intermediate_z"][0]; mlp_z = corrected_zs["mlp_z"][0]; hidden_z = corrected_zs["hidden_z"]
        
        mlp1, mlp2 = copy_module()
        mlp1.prune_params({"intermediate_z": intermediate_z, "mlp_z": mlp_z, "hidden_z": hidden_z})
        
        hidden_states = hidden_states.mul(hidden_z)
        compressed_hidden_states = hidden_states.index_select(2, torch.where(~hidden_z.eq(0))[0])
        
        x1 = mlp1(compressed_hidden_states)
        if x1 is not None:
            x1 = x1.sum()
        x2 = mlp2(hidden_states, intermediate_z=intermediate_z, mlp_z=mlp_z, hidden_z=hidden_z).sum()
        return x1, x2
    
    def eval(v1, v2, case_num=0):
        nice_print(v1, v2)
        if torch.isclose(v1, v2):
            print(f"case {case_num} passed!")
        else:
            print(f"case {case_num} failed!")
             
    # case 1: 
    print("\n[Test MLP] case 1: all intermediate dims are pruned")
    percentage = {"intermediate_z": 0.3, "mlp_z": 1.} 
    x1, x2 = run(percentage)
    if x1 is None and x2 == .0:
        print("case 1 passed!")
    else:
        nice_print(x1, x2)
        print("case 1 failed!")
    
    # case 2: 
    print("\n[Test MLP] case 2: a non-zero number of intermediate dims are pruned")
    percentage = {"intermediate_z": 0.3} 
    x1, x2 = run(percentage)
    eval(x1, x2, 2) 
    
    # case 3: 
    print("\n[Test MLP] case 3: a non-zero number of hidden_dims are pruned")
    percentage = {"hidden_z": 0.3} 
    x1, x2 = run(percentage)
    eval(x1, x2, 3)
     
    # case 4: 
    print("\n[Test MLP] case 4: a non-zero number of intermediate dims are pruned and hidden_dims are pruned")
    percentage = {"intermediate_z": 0.3, "hidden_z": 0.3} 
    x1, x2 = run(percentage)
    eval(x1, x2, 4)
     
    
# passed
def test_Shearing_decode_layer(model, l0_module, half=False, ones=False):
    zs = get_full_zs(l0_module, half=half, ones=ones)
    layer_num = 5
    layer = deepcopy(model.model.transformer.blocks[layer_num])
    device = next(model.parameters()).device
    
    def copy_module():
        layer1 = deepcopy(layer).cuda()
        layer2 = deepcopy(layer).cuda()
        if half:
            layer1 = layer1.half()
            layer2 = layer2.half()
        layer1.eval(); layer2.eval()
        return layer1, layer2
    
    def init(percentage):
        corrected_zs = zero_out_all_zs(deepcopy(zs), percentage)
        zs_block = {}
        for key in percentage:
            if key == "hidden_z":
                zs_block[key] = corrected_zs["hidden_z"]
            else:
                zs_block[key] = corrected_zs[key][layer_num]
        return zs_block
                
    def execute(zs_block):
        layer1, layer2 = copy_module(); layer1.prune_params(zs_block)
        with torch.no_grad():
            hidden_states = torch.randn(2, 3, len(layer2.ln_1.weight)).to(device); 
            if half: hidden_states = hidden_states.half()
            pruned_hidden_states = hidden_states
            hidden_z = zs_block.get("hidden_z", None)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z != 0)
                pruned_hidden_states = hidden_states[..., hidden_z.squeeze().nonzero().squeeze()]
            else:
                pruned_hidden_states = hidden_states 
            layer_output1 = layer1(pruned_hidden_states)[0]
            layer_output2 = layer2(hidden_states, **zs_block)[0]
            
            v2 = layer_output2.sum()
            if layer_output1 is not None: v1 = layer_output1.sum(); 
            else: v1 = torch.zeros_like(v2) 
        return v1, v2
                
    # case 1
    print("\n[Test layer] case 1: Some heads are pruned and some intermediate dims are pruned")
    percentage = {"head_z": 0.3, "intermediate_z": 0.3}
    zs_block = init(percentage)
    v1, v2 = execute(zs_block)    
    eval(v1, v2, 1)

    # case 2
    print("\n[Test layer] case 2: A few hidden dims are pruned")
    percentage = {"hidden_z": 0.3, "head_z": 1., "intermediate_z": 1.}
    zs_block = init(percentage)
    v1, v2 = execute(zs_block)    
    eval(v1, v2, 2)
    
    # case 3
    print("\n[Test layer] case 3: some heads/intermediate dims/hidden dims are pruned")
    percentage = {"hidden_z": 0.3, "head_z": 0.3, "intermediate_z": 0.3}
    zs_block = init(percentage)
    v1, v2 = execute(zs_block)    
    eval(v1, v2, 3)

# passed
def test_Shearing_llama_model(model, l0_module, half, ones=False):
    zs = get_full_zs(l0_module, half=half, ones=ones)
    input_ids = load_input_ids(cuda=True)
    
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0.3, "intermediate_z": 0.3, "mlp_z": 0.4, "head_layer_z": 0.5, "hidden_z": 0.6})
    
    model1 = deepcopy(model).cuda()
    model2 = deepcopy(model).cuda()
    if half:
        model1 = model1.half()
        model2 = model2.half()
    model1.prune_params(corrected_zs)

    output1 = forward(model1, input_ids, {})
    output2 = forward(model2, input_ids, corrected_zs)
    if torch.isclose(output1.sum(), output2.sum()):
        print("test_prune_opt_model passed!")
    else:
        print("v1: ", output1.sum())
        print("v2: ", output2.sum())
        print("test_prune_opt_model failed!")

if __name__ == "__main__":
    # retest after setting get_full_zs: ones=True 
    cfg = construct_example_cfg("7B", True) 
    cfg.l0_module.pruning_modules = ["layer", "head", "intermediate", "hidden"]
    cfg.path = "/projects/DANQIC/mengzhou/LLaMA/mosaic-7B/state_dict.pt"
    model = build_composer_model(cfg).cuda()
    l0_module = model.model.l0_module
    model.model.l0_module = None
    
    model.train()
    l0_module.train()
    
    ones = False 
    test_Shearing_LayerNorm(l0_module)
    test_full_z(model, l0_module, half=True, ones=ones)
    test_Shearing_Attention(model, l0_module, half=True, ones=ones)
    test_Shearing_MLP(model, l0_module, half=True, ones=ones)
    test_Shearing_decode_layer(model, l0_module, half=True, ones=ones)
    test_Shearing_llama_model(model, l0_module, half=True, ones=ones)
    # from hf_llama.tokenization_llama import LlamaTokenizer
    # tokenizer = LlamaTokenizer.from_pretrained("/scratch/gpfs/mengzhou/LLaMA/hf-7B")
    # import pdb; pdb.set_trace()


