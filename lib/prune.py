import time 
import torch 
import torch.nn as nn 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import numpy as np
import torch.nn as nn

from pdb import set_trace as st 
# from .quant import *
import numpy as np
from scipy.optimize import linear_sum_assignment

def lexsort(keys, dim=-1):
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx


def maximize_total_value(matrix):
    row_indices, col_indices = linear_sum_assignment(matrix, maximize=True) 
    return col_indices


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(args, model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in args.model:
        layers = model.model.layers
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]
    elif "opt" in args.model:
        layers = model.model.decoder.layers


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, args, module):
            super().__init__()
            self.module = module
            self.model = args.model
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model:
                cache['position_ids'] = kwargs['position_ids']

            raise ValueError
    layers[0] = Catcher(args, layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
   
    model.config.use_cache = use_cache
    if "llama" in args.model:
        position_ids = cache['position_ids']
        return inps, outs, attention_mask, position_ids 
    elif "opt" in args.model:
        return inps, outs, attention_mask



def z_pruner(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.calib_dataset,nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "llama" in args.model:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
        elif "opt" in args.model:
            inps, outs, attention_mask= prepare_calibration_input(args, model, dataloader, device)
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:  
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)
            if args.gptq:
                wrapped_layers[name].quantizer = Quantizer()
                wrapped_layers[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()
        for name in subset:
            if args.gptq:
                wrapped_layers[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
            W = subset[name].weight.data.clone()
            
            def compute_importance_scores_tuned(W):
                  
                  norm_row = torch.norm(W, p=2, dim=1, keepdim=True)
                  W_norm_row = W / (norm_row)  

                  norm_col = torch.norm(W, p=2, dim=0, keepdim=True)
                  W_norm_col = W / (norm_col)  

                  mu_row = torch.mean(W_norm_row)
                  sigma_row = torch.std(W_norm_row)

                  mu_col = torch.mean(W_norm_col)
                  sigma_col = torch.std(W_norm_col)

                  diff_row = (W_norm_row - mu_row) / (sigma_row)
                  diff_col = (W_norm_col - mu_col) / (sigma_col)
                  
                  importance_row = torch.abs(diff_row**2) * torch.abs(diff_row)
                  importance_col = torch.abs(diff_col**2) * torch.abs(diff_col)

                  sparsity = (W.abs() < W.abs().mean() * 0.1).float().mean()
                  alpha = 0.7 * (1 - 0.3 * sparsity) 
                  combined_importance = alpha * importance_row + (1 - alpha) * importance_col

                  return combined_importance




            def root_tanh_scaling(x, beta=0.7, gamma=2.5, alpha=1.0):
                  return alpha * (torch.tanh(torch.abs(x) ** gamma)) * (torch.abs(x) ** beta)
            


            activation = root_tanh_scaling(wrapped_layers[name].scaler_row.reshape((1, -1)))
            activation2 = (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**1.5

            W_metric = compute_importance_scores_tuned(W) * (activation2)

            W_mask = (torch.zeros_like(W_metric) == 1) 
            
            if args.per_outneuron:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)
                
            if args.reconstruction:
                wrapped_layers[name].fasterprune(args.sparsity_ratio, mask=W_mask)
            else:
                subset[name].weight.data[W_mask] = 0  
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

                   
    model.config.use_cache = use_cache