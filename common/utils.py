import torch

def recursive_to(input):
    if isinstance(input, torch.Tensor):
        return input.cuda()
    if isinstance(input, dict):
        for name in input:
            if isinstance(input[name], torch.Tensor):
                input[name] = input[name].cuda()
        return input
    if isinstance(input, list):
        for i, item in enumerate(input):
            input[i] = recursive_to(item)
        return input
    assert False