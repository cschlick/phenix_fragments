import torch

def to_np(tensor):
  return tensor.detach().cpu().numpy()


def to_torch(array):
  return torch.tensor(array,dtype=torch.get_default_dtype())