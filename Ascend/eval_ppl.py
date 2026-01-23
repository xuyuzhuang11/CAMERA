import torch

@torch.no_grad()
def ppl_eval(model, testenc, dev):
    print('Evaluating ...')
    
    testenc = testenc.input_ids
    testenc = testenc.to(dev)
    nsamples = testenc.numel() // model.seqlen
    
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        output = model(batch)
        shift_logits = output.logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    

@torch.no_grad()
def pangu_eval_npu(model_path):
    from transformers import AutoModelForCausalLM
    from datautils import get_loaders

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="npu",
        local_files_only=True
    )

    model.seqlen = 2048
    _, testloader_wikitext2 = get_loaders(
        'wikitext2', nsamples=16, seed=1234, model=model_path, seqlen=model.seqlen
    )
    _, testloader_c4 = get_loaders(
        'c4', nsamples=16, seed=1234, model=model_path, seqlen=model.seqlen
    )

    print("wikitext: ")
    ppl_eval(model, testloader_wikitext2, model.device)
    print("c4: ")
    ppl_eval(model, testloader_c4, model.device)


@torch.no_grad()
def pangu_pruned_eval_npu(model_path, pruned_ckpt):
    from transformers import AutoModelForCausalLM
    from datautils import get_loaders

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="npu:0",
        local_files_only=True
    )
    checkpoint = torch.load(pruned_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint)

    model.seqlen = 2048
    _, testloader_wikitext2 = get_loaders(
        'wikitext2', nsamples=16, seed=1234, model=model_path, seqlen=model.seqlen
    )
    _, testloader_c4 = get_loaders(
        'c4', nsamples=16, seed=1234, model=model_path, seqlen=model.seqlen
    )

    print("wikitext: ")
    ppl_eval(model, testloader_wikitext2, model.device)
    print("c4: ")
    ppl_eval(model, testloader_c4, model.device)


if __name__ == "__main__":
    # # PPL (Pruned openPangu-Embedded-7B-V1.1)
    # model_path = "/opt/pangu/openPangu-Embedded-7B-V1.1"
    # pruned_ckpt = "/home/lpzhan/data/model/camera/pangu_7B_pruned_0.5_reduce_0.9/prune_model_dict.pt"
    # pangu_pruned_eval_npu(model_path, pruned_ckpt)

    # PPL (openPangu-Embedded-7B-V1.1)
    model_path = "/opt/pangu/openPangu-Embedded-7B-V1.1/"
    pangu_eval_npu(model_path)
