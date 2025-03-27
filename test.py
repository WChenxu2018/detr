import torch
src_logits = torch.tensor([
    [[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0], [1.0, 3.0, 2.0], [2.0, 1.0, 3.0]],
    [[2.0, 3.0, 1.0], [3.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0]]
])
print(src_logits.transpose(1,2))
print(src_logits.transpose(1,2).size())