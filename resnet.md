# Deep Residual Learning for Image Recognition
## Key Points
### x(input) + f(x)(Residual Function) = H(x)(output)
* if identity mapping is optimal, x and H(x) are similar so that f(x) would be zero.
* it means multi-nonliner weights would be zero.
* as layer gets deeper, residual function has few trains, therefore it trains similar to identity function, called H(x)

### Train
in forward propagation, it trains h(x)[f(x)+x], and in backward propagation, it trains to reduce the loss function to fit f(x) toward zero.

###### class ResidualBlock(nn.Module):
######    def __init__(self):
######        super(ResidualBlock.self)__init__()
######        self.linear1 = nn.Linear(10,10)
######        self.linear2 = nn.Linear(10,10)
######    def forward(self, x): 
######        residual = self.linear1(x)
######        residual = nn.ReLU()(residual)
######        residual = self.linear2(residual)
######        # skip connection
######        out = residual + x
######        return out
######
###### res_block = ResidualBlock()   
###### loss_function = nn.MSELoss()
###### optimizer = torch.optim.SGD(res_block.parameters(), lr=0.01)
###### target = torch.rand(10)
###### output = res_block(x)
###### loss = loss_function(output, target)
###### optimizer.zero_grad()
###### loss.backward() 
* calculate network weights gradient about loss function
* compute weight loss gradient from H(x) backwards to input x so that f(x) would be zero.
  

### How to resolve vanishing gradient, overfitting, computation increase?
use residual function to not add a parameters if x and f(x) dimension is same.
#### What is parameters?
the variants to be learned. 
to train pattern from data, we mediate inner variants, W(weight) and B(bias)
in this paper, they only put parameters when x and f(x) has different dimensions to fit it.
so compared to vgg16, they increased the mAP from lower parameters

###### class LinearLayer(nn.Module):
######    def __init__(self, input_dim, output_dim):
######        super(LinearLayer).__init__()
######        self.linear = nn.Linear(input_dim, output_dim)
######    def forward(self, x):
######        return self.linear(x)
######        
###### input_vector = torch.tensor([[1.0,2.0]])        
###### linear_layer = LinearLayer(input_dim=2, output_dim=3)
###### output_vector = linear_layer(input_vector)

###### Input Vector: tensor([[1., 2.]])
###### Output Vector: tensor([[0.5095, 1.2703, 0.2823]], grad_fn=<AddmmBackward0>)
###### Linear Layer Parameters: [Parameter containing:
###### tensor([[ 0.0751,  0.4416],
######         [-0.3733,  0.5463],
######         [ 0.5653,  0.0331]], requires_grad=True), Parameter containing:
###### tensor([-0.4488,  0.5510, -0.3492], requires_grad=True)]
