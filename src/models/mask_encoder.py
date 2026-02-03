import torch
import torch.nn as nn
class MaskEncoder(nn.Module):
    def __init__(self,in_ch = 1, out_ch = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,  padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3 , padding = 1)
        )
    def forward(self,mask):
        return self.net(mask)