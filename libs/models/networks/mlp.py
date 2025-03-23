import torch.nn as nn


class basic_project2(nn.Module):
    def __init__(self, input_ch, output_ch, bias=True):
        super(basic_project2, self).__init__()
        self.proj = nn.Linear(input_ch, output_ch, bias=bias)
    def forward(self, x):
        return self.proj(x)

class kernel_linear_act(nn.Module):
    def __init__(self, input_ch, output_ch, bias=True):
        super(kernel_linear_act, self).__init__()
        self.block = nn.Sequential( 
            nn.PReLU(),
            basic_project2(input_ch, output_ch, bias))
    def forward(self, input_x):
        return self.block(input_x)

