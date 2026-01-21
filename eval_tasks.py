from lm_eval import tasks, evaluator
import json
from datetime import datetime
import os
import torch
import numpy as np
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import Qwen3MoeForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig, MixtralForCausalLM
# from .model.modeling_phimoe import PhiMoEForCausalLM, PhiMoEConfig


def run_lm_eval(model, tokenizer, batch_size=16, task_names=["boolq", "openbookqa", "rte", "arc_easy", "winogrande", "hellaswag", "arc_challenge", "piqa", "mathqa"], output_dir=""):
    # Import the correct task loading function

    results = evaluator.simple_evaluate(
        model=model,
        tokenizer=tokenizer,
        tasks=task_names,
        batch_size=batch_size,
        device=next(model.parameters()).device,
        write_out=True,
        log_samples=True,
        verbosity="INFO",
        num_fewshot=0,
        task_manager=tasks.TaskManager(),
    )

    # Remove samples from results to reduce file size
    if 'samples' in results:
        del results['samples']

    # Custom JSON Encoder to handle torch.Tensor, torch.device, and numpy.ndarray
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, torch.device):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"results_{timestamp}.json")

    # Save the results dictionary to a JSON file using the custom encoder
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, cls=CustomEncoder)

    print(f"Results saved to {filename}")
    

if __name__ == '__main__':

    model_path = '/home/share/models/deepseek-moe-16b-base'
    # model_path = '/home/share/models/Qwen3-30B-A3B'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    no_split_module_classes = ["Qwen3MoeDecoderLayer"]
    max_memory = {i: "65GiB" for i in range(torch.cuda.device_count())}   # set the max memory for each GPU
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
    model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map, dtype=torch.bfloat16)
    
    checkpoint = torch.load('/home/yzxu/MoE/d16b/0.21_energycut_tasktest/prune_model_dict.pt')
    model.load_state_dict(checkpoint)
    torch.cuda.empty_cache()
    
    save_dir = '/home/yzxu/MoE/d16b/0.21_energycut_tasktest'
    run_lm_eval(model, tokenizer, batch_size=16, output_dir=save_dir)
