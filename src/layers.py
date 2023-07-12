import torch.nn as nn 
import torch 
import numpy as np 

class Reshape(nn.Module):
    def __init__(self, img_shape:tuple[int]) -> None:
        super().__init__()
        self.img_shape = img_shape

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return x.view((x.shape[0], *self.img_shape))
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_img_shape={}'.format(
            np.prod(self.img_shape),self.img_shape,
        )