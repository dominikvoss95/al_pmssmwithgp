import gpytorch
import torch

class LinearMean(gpytorch.means.Mean):
    
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(
            name="weights",
            parameter=torch.nn.Parameter(0.1 * torch.ones(*batch_shape, input_size, 1)),
        )
        if bias:
            self.register_parameter(
                name="bias", parameter=torch.nn.Parameter(torch.ones(*batch_shape, 1))
            )
        else:
            self.bias = None
    
    def forward(self, x):
        # print("weights:", self.weights.shape)
        # print("X:", x.shape)
        # res = x.matmul(self.weights).squeeze(-1) # Making it a 2D Tensor
        res = x.to(self.weights.device).matmul(self.weights).squeeze(-1)
        # print("X:", x)
        # print("weights:", self.weights)
        if self.bias is not None:
            res = res + self.bias
        return res
    
