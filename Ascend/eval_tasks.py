from lm_eval import tasks, evaluator
import json
from datetime import datetime
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
    

def pangu_pruned_lm_eval():
    """
    Eval (Pruned openPangu-Embedded-7B-V1.1)
    """
    from lm_eval.models.huggingface import HFLM
    model_path = "/opt/pangu/openPangu-Embedded-7B-V1.1"
    pruned_ckpt = "/home/lpzhan/data/model/camera/pangu_7B_pruned_0.5_reduce_0.9/prune_model_dict.pt"

    lm = HFLM(
        pretrained=model_path,
        trust_remote_code=True,
        device="npu:1",
        batch_size=64,
        dtype="float16",
        local_files_only=True,
    )
    checkpoint = torch.load(pruned_ckpt, map_location="cpu") 
    lm.model.load_state_dict(checkpoint)

    tasks = ["boolq", "openbookqa", "rte", "arc_easy", "winogrande", "hellaswag", "arc_challenge", "piqa", "mathqa"]

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=64,
        num_fewshot=0,
        verbosity="WARNING",
    )

    # Remove samples from results to reduce file size
    if 'samples' in results:
        del results['samples']

    # Custom JSON Encoder to handle torch.Tensor, torch.device, and numpy.ndarray
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            print(type(obj), obj)
            # 1. 处理 torch.dtype
            if isinstance(obj, torch.dtype):
                return str(obj)
            # 2. 处理 numpy.dtype
            if isinstance(obj, np.dtype):
                return str(obj)
            # 3. 处理类型对象（如 <class 'numpy.float32'>）
            if isinstance(obj, type):
                return str(obj)
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
    output_dir = "/home/lpzhan/data/prune/camera/out/"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"results_{timestamp}.json")

    # Save the results dictionary to a JSON file using the custom encoder
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, cls=CustomEncoder)

    print(f"Results saved to {filename}")


def pangu_lm_eval():
    """
    Eval (openPangu-Embedded-7B-V1.1)
    """
    from lm_eval.models.huggingface import HFLM
    model_path = "/opt/pangu/openPangu-Embedded-7B-V1.1"

    lm = HFLM(
        pretrained=model_path,
        trust_remote_code=True,
        device="npu:1",
        batch_size=64,
        dtype="float16",
        local_files_only=True,
    )

    tasks = ["boolq", "openbookqa", "rte", "arc_easy", "winogrande", "hellaswag", "arc_challenge", "piqa", "mathqa"]

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=64,
        num_fewshot=0,
        verbosity="WARNING",
    )

    # Remove samples from results to reduce file size
    if 'samples' in results:
        del results['samples']

    # Custom JSON Encoder to handle torch.Tensor, torch.device, and numpy.ndarray
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            print(type(obj), obj)
            # 1. 处理 torch.dtype
            if isinstance(obj, torch.dtype):
                return str(obj)
            # 2. 处理 numpy.dtype
            if isinstance(obj, np.dtype):
                return str(obj)
            # 3. 处理类型对象（如 <class 'numpy.float32'>）
            if isinstance(obj, type):
                return str(obj)
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
    output_dir = "/home/lpzhan/data/prune/camera/out/"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"results_{timestamp}.json")

    # Save the results dictionary to a JSON file using the custom encoder
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, cls=CustomEncoder)

    print(f"Results saved to {filename}")


if __name__ == '__main__':
    # pangu_pruned_lm_eval()
    pangu_lm_eval()
