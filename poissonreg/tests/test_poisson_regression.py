import unittest
import torch
from poissonreg.regression import generate_sample, PoissonRegression, train

torch.manual_seed(1)
class TestPoissonRegression(unittest.TestCase):

	def test_poisson_regression(self):
		data = generate_sample(100000, 2)
		threshold = 1e-10
		X, Y, theta = data["X"], data["Y"], torch.flatten(data["theta"])
		W = torch.ones(Y.shape)
		model = PoissonRegression(X.shape[1])
		losses = train(model, X.float(), Y.float(), W.float(), threshold = 1e-20, max_steps = 2000)		
		estimated = model.theta.weight
		max_grad = estimated.grad.abs().max()
		self.assertLess((estimated - theta).abs().max(), 5e-2, msg="weights are not close to actual parameters")
		self.assertLess(max_grad, threshold, msg="maximum grad is too large")