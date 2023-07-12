import os
import random
import numpy as np 
import torch

__all__ = [
    "set_seed",
    ]

def set_seed(seed:int=42)->None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True