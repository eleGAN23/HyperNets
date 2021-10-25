import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, init
from torch.nn.parameter import Parameter

########################
## STANDARD PHM LAYER ##
########################

class PHMLinear(nn.Module):

  def __init__(self, n, in_features, out_features, cuda=True):
    super(PHMLinear, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))

    self.a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))

    self.s = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features//n, self.in_features//n))))

    self.weight = torch.zeros((self.out_features, self.in_features))

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)


  def kronecker_product1(self, a, b): #adapted from Bayer Research's implementation
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features))
    for i in range(self.n):
        H = H + torch.kron(self.a[i], self.s[i])
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
#     self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())
    return F.linear(input, weight=self.weight, bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.a, a=math.sqrt(5))
    init.kaiming_uniform_(self.s, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

################################
## PHC LAYER: 2D convolutions ##
################################

class PHMConv2d(Module):

  def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, cuda=True):
    super(PHMConv2d, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.padding = padding
    self.stride = stride
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))
    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, kernel_size, kernel_size))))
    self.weight = torch.zeros((self.out_features, self.in_features))
    self.kernel_size = kernel_size

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-4:-2]))
    siz2 = torch.Size(torch.tensor(F.shape[-2:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-4).unsqueeze(-6)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
    if self.cuda:
        H = H.cuda()
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
    if self.cuda:
        self.weight = self.weight.cuda()

    input = input.type(dtype=self.weight.type())
        
    return F.conv2d(input, weight=self.weight, stride=self.stride, padding=self.padding)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

################################
## PHC LAYER: 1D convolutions ##
################################
    
class PHConv1D(Module):

  def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, dilation=1, cuda=True):
    super(PHConv1D, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.padding = padding
    self.stride = stride
    self.dilation=dilation
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))

    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, kernel_size))))
    self.weight = torch.zeros((self.out_features, self.in_features))
    self.kernel_size = kernel_size

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-3:-1]))
    siz2 = torch.Size(torch.tensor(F.shape[-1:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1) * F.unsqueeze(-3).unsqueeze(-5)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
    if self.cuda:
        H = H.cuda()
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
    if self.cuda:
        self.weight = self.weight.cuda()

    input = input.type(dtype=self.weight.type())
    
      def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)
        
    return F.conv1d(input, weight=self.weight, stride=self.stride, padding=self.padding,dilation=self.dilation)
