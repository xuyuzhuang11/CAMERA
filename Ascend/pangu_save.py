import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def save_logical_pruned_model(origin_path, pruned_ckpt, dst_path):
    model = AutoModelForCausalLM.from_pretrained(
        origin_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(origin_path, trust_remote_code=True, use_fast=False)
    checkpoint = torch.load(pruned_ckpt, map_location="cpu") 
    model.load_state_dict(checkpoint)
    model = model.cpu()

    # Save in HF format (with config, tokenizer, weight)
    model.save_pretrained(dst_path, safe_serialization=True)    # weight and config
    tokenizer.save_pretrained(dst_path)                         # tokenizer

def save_physical_pruned_model(origin_path, pruned_ckpt, dst_path, origin_inter_size, prune_ratio):
    """
    Args:
        origin_path: origin model path (HF)
        pruned_ckpt: physical_pruned ckpt
        dst_path: dist model path (HF)
        origin_inter_size: original 'config.intermediate_size'
        prune_ratio: ratio
    """
    print(f"1. Loading config from {origin_path}...")

    config = AutoConfig.from_pretrained(origin_path, trust_remote_code=True)
    
    # Change the config.intermediate_size to the pruned size
    original_size = config.intermediate_size
    target_size = origin_inter_size * prune_ratio   # pangu-7B: (12800) * prune_ratio(0.8) = 10240
    print(f"   Modifying intermediate_size: {original_size} -> {target_size}")
    config.intermediate_size = target_size
    
    # Create the physical pruned model structure without loading weight
    print("2. Initializing model structure with new config...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Load the physical pruned weight
    print(f"3. Loading pruned weights from {pruned_ckpt}...")
    checkpoint = torch.load(pruned_ckpt, map_location="cpu")
    
    # A simple shape checking
    try:
        msg = model.load_state_dict(checkpoint, strict=True)
        print("   Load state dict result:", msg)
    except RuntimeError as e:
        print("!! Error loading state dict. Please check if keys match.")
        print("Common error: Size mismatch implies config.intermediate_size is wrong.")
        raise e

    # Saving the model
    print(f"4. Saving to {dst_path}...")
    model.save_pretrained(dst_path, safe_serialization=True)
    
    # Saving the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(origin_path, trust_remote_code=True, use_fast=False)
    tokenizer.save_pretrained(dst_path)
    
    print("Done.")


def load_test(model_path):
    tok  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl  = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="npu:1",
        local_files_only=True
    )
    print("Loaded!")

def load_test_physical_prune(model_path):
    print(f"Testing load from {model_path}...")
    # Now the intermediate_size in config.json has been modified, the from_pretrained will work well.
    tok  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl  = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="npu:1", # or "cuda:0" / "cpu"
        local_files_only=True
    )
    print("Loaded successfully! Check MLP size:")
    # Check the shape to make sure the model has been physical pruned.
    try:
        if hasattr(mdl, 'model') and hasattr(mdl.model, 'layers'):
            layer0_mlp = mdl.model.layers[0].mlp
            print(f"Up proj shape: {layer0_mlp.up_proj.weight.shape}") 
            # should print [10240, 4096] (assuming hidden_size=4096)
    except:
        print("Could not verify specific layer shape, but model loaded.")
    

if __name__ == "__main__":
    origin_path = "/home/model/openPangu-Embedded-7B-V1.1/"
    pruned_ckpt = "/home/model/camera/pangu_7B_pruned_0.8_reduce_0.95_physical/prune_model_dict.pt"
    dst_path = "/home/model/camera/pangu_7B_pruned_0.8_reduce_0.95_physical/openPangu-Embedded-7B-PysicalPruned/"
    
    # # Save logic_pruned model
    # save_logical_pruned_model(origin_path, pruned_ckpt, dst_path)
    # load_test(dst_path)

    # Save physical_pruned model
    save_physical_pruned_model(origin_path, pruned_ckpt, dst_path)
    load_test_physical_prune(dst_path)
    
