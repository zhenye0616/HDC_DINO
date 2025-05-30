import math
import torch


class Encoder(object):
    def __init__(self, features : int, dim : int = 4000):
        self.dim = dim
        self.features = features
        self.basis = torch.randn(self.dim, self.features)
        self.base = torch.empty(self.dim).uniform_(0.0, 2*math.pi)

    def __call__(self, x : torch.Tensor):
        #import pdb;pdb.set_trace()
        # print("Input shape:", x.shape)  # Debug print
        # print("Expected features:", self.features)
        n = x.size(0)       # batch size, e.g., 4
        T = x.size(1)       # number of tokens, e.g., 19947
        bsize = math.ceil(0.01 * n)  # batch chunk size; for n=4, bsize becomes 1
        # Allocate output tensor h with shape [n, T, self.dim]
        h = torch.empty(n, T, self.dim, device=x.device, dtype=x.dtype)
        # Allocate temporary tensor temp with shape [bsize, T, self.dim]
        temp = torch.empty(bsize, T, self.dim, device=x.device, dtype=x.dtype)

        for i in range(0, n, bsize):
            #import pdb;pdb.set_trace()
            # print("Multiplying x[{}:{}] shape:".format(i, i+bsize), x[i:i+bsize].shape)
            # print("self.basis.T shape:", self.basis.T.shape)
            #torch.matmul(x[i:i+bsize], self.basis.T, out=temp)
            temp = torch.matmul(x[i:i+bsize], self.basis.T)
            h_chunk = torch.add(temp, self.base)
            # Apply cos and sin operations; note that these operations are not in-place here
            h_chunk = h_chunk.cos() * temp.sin()
            # Store the result in the preallocated h tensor
            h[i:i+bsize].copy_(h_chunk)
        return h
    

    def to(self, *args):
        self.basis = self.basis.to(*args)
        self.base = self.base.to(*args)
        return self