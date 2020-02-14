import sys
import numpy as np
import torch
import torch.nn as nn

class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(7, 64 , kernel_size = 45, padding = 22, bias=True), 
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128,  kernel_size = 25, padding = 12,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256 , kernel_size = 15, padding = 7, bias=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.ReLU(inplace=True),  
        )
        
    def forward(self, input):
        return self.main(input)