import torch
from tqdm import tqdm
import numpy as np

def generate_sample(n=20000, k=4, density = 0.1):
    """
generates a sample from poissons where poisson's parameter is random with lambda=exp(X@theta)

Args:
    n: int, default 20000
        number of observations
    k: int, default 4
        number of variables per observation
    density: float, default 0.1
        fraction of X's cells that will be set to zero by random, controls the sparsity

Returns:
    Returns a dictionary containing
        'n': int
            number of observations
        'k': int
            number of variables per observation
        'X':torch.tensor
            The independant
        'theta': torch.tensor
            Actual weights assigned to independant variables in making of poisson parameters
        'lambdas': torch.tensor
            Actual weights with which poisson parameters are constructed
        'Y': torch.tensor
            The dependant
"""
    X = torch.rand(n,k)
    mask = torch.rand(n,k) < density
    X[mask] = 0
    theta = torch.rand(k, 1)
    lambdas = torch.exp(X @ theta)
    Y = torch.poisson(lambdas)
    return {
        "n": n,
        "k": k,
        "X": X.to_sparse(),
        "theta": theta,
        "lambdas": lambdas,
        "Y": Y,
    }

def train(model, X, Y, W, max_steps=200, optimizer = None, verbose = True, threshold=1e-11):
    """
Trains the model on dependant/independant variables.
Args:
    model: torch.nn.Module
        The model that's going to be trained
    X: torch.tensor
        The n*k independant variable
    Y: tensor.torch
        The flat dependant variable
    W: tensor.torch
        The flat weights of samples
    max_steps: int, default 200
        Maximum number of steps to take for the optimization.
    optimizer: torch.optim.Optimizer, default None
        The optimizer used for training. If not provided, torch.optim.RMSprop will be used.
    verbose: bool, default True
        Determines whether a progress bar for training is printed.
    threshold: float, 0.default 000001
        Optimization is terminated when loss is changed less than threshold between two consecutive steps.

Returns: list(float)
    a list of losses obtained during the optimization
"""
    if optimizer is None:
        optimizer = torch.optim.RMSprop(model.parameters())
    losses = [torch.nan]
    for i in tqdm(range(max_steps), disable = not verbose):
        optimizer.zero_grad()
        loss = model.get_loss(X, Y, W)
        loss.backward()
        max_grad_size = np.max([p.grad.abs().max().item() for p in model.parameters()])
        if max_grad_size<threshold:
            if verbose:
                print(f"success after {i} steps")
            break
        losses.append(loss)        
        optimizer.step()
    return losses

class PoissonRegression(torch.nn.Module):
    """A model for estimating
    Args:
        size: int
            number of variables of the independant variable this model will be used on
        intercept: bool, default False
            Whether the linear layer has an intercept
        l1_weight: float, default 0.
            Weight assigned to l1 penalization in loss
        l2_weight: float, default 0.
            Weight assigned to l2 penalization in loss
    Attributes:
        size: int
            number of variables of the independant variable this model will be used on
        theta: torch.nn.Linear
            Models parameters consisting of a size*1 layer with intercept
        l1_weight: float
            Weight assigned to l1 penalization in loss
        l2_weight: float
            Weight assigned to l2 penalization in loss
    """
    def __init__(self, size, intercept=False, l1_weight=0., l2_weight=0.):
        super(PoissonRegression, self).__init__()
        self.size = size
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.theta = torch.nn.Linear(size, 1, intercept)
    
    def forward(self, x):
        linear = self.theta.forward(x)
        return torch.exp(linear)
    
    def get_loss(self, X, Y, W):
        """Computes model's weighted loss on dependant and independant variables
        Args:
            X: torch.tensor
                Independant Variable
            Y: torch.tensor
                Dependant Variable
            W: torch.tensor
                Weights
        """
        W = W / W.sum()
        weighted_Y = Y * W
        log_likelihood = weighted_Y.T @ self.theta.forward(X) - W.T @ self.forward(X)
        params = torch.nn.utils.parameters_to_vector(self.parameters())
        l1 = self.l1_weight * torch.norm(params, 1)
        l2 = self.l2_weight * torch.norm(params, 2)
        loss = -log_likelihood
        return loss + l1 + l2