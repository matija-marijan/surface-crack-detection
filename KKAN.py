import os
import sys
from torch import nn
import torch.nn.functional as F

from ConvKAN.kan_convolutional.KANConv import KAN_Convolutional_Layer
from ConvKAN.kan_convolutional.KANLinear import KANLinear

# library_path = '/home/matijamarijan/projects/ConvKAN'
# sys.path.append(library_path)
# sys.path.append('../kan_convolutional')

# from ..ConvKAN.kan_convolutional
# from ConvKAN.kan_convolutional
# from ConvKAN.kan_convolutional import 

class KKAN(nn.Module):
   def __init__(self, device: str = 'cpu'):
      super().__init__()

      self.conv1 = KAN_Convolutional_Layer(
         n_convs = 5,
         kernel_size= (3,3),
         device = device
      )

      self.conv2 = KAN_Convolutional_Layer(
         n_convs = 5,
         kernel_size= (3,3),
         device = device
      )

      self.conv3 = KAN_Convolutional_Layer(
         n_convs = 5,
         kernel_size= (3,3),
         device = device
      )

      self.pool1 = nn.MaxPool2d(
         kernel_size = (2, 2)
      )
        
      self.flat = nn.Flatten()

      self.kan1 = KANLinear(
         # 84500,
         4500,
         2
      )

   def forward(self, x):
      x = self.conv1(x)
      x = self.pool1(x)

      x = self.conv2(x)
      x = self.pool1(x)

      x = self.conv3(x)
      x = self.pool1(x)

      x = self.flat(x)

      x = self.kan1(x) 
      x = F.log_softmax(x, dim=1)

      return x
