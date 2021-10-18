import torch
from torch import nn
import torch.nn.functional as F

class myCELU(nn.Module):
    """ 
    klon torch.nn.modules.activation.CELU bo ONNX wysypuje się na oryginale
    """
    
    __constants__ = ['alpha', 'bound']

    def __init__(self, alpha=1.):
        super(myCELU, self).__init__()
        self.alpha = alpha
        self.bound = 30

    def forward(self, input):
        # return F.celu(input, self.alpha, self.inplace)
        # return torch.celu(input, self.alpha)
        # return torch.relu(input)
        # max(0,x)+min(0,α∗(exp(x/α)−1))
        zeros = torch.zeros(input.shape)
        
        inp_alpha = input / self.alpha

        #if (inp_alpha > self.bound).any():
        if torch.max(input) > self.bound:
            maxs = torch.zeros(input.shape) + self.bound
            inp_alpha = torch.where(inp_alpha < maxs, inp_alpha, maxs)
            
        #if (inp_alpha < -self.bound).any():
        if torch.min(input) < -self.bound:
            mins = torch.zeros(input.shape) - self.bound
            inp_alpha = torch.where(inp_alpha > mins, inp_alpha, mins)

        if (inp_alpha != inp_alpha).any():
            print(inp_alpha)
        # (torch.min(input).item(), torch.max(input).item())    
            
        return torch.max(zeros, input) + torch.min(zeros, self.alpha * (torch.exp(inp_alpha) - 1) ) 
