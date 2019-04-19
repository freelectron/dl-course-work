from random import random as generate_random

import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""


######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
    """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormAutograd object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomBatchNormAutograd, self).__init__()

        # intit the variables. see also nn.BatchNorm2d() might be useful

        # number of neurons C (channels). holy shit why cant we all just name things in the same manner
        # the world would be so much better.
        self.n_neurons = n_neurons
        self.eps = eps
        # this is not tested ...                                                                                  
        self.gamma = nn.Parameter(torch.ones(self.n_neurons))
        self.beta = nn.Parameter(torch.zeros(self.n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor

        TODO:
          Check for the correctness of the shape of the input tensor.
          Implement batch normalization forward pass as given in the assignment.
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.

        ---- Additional info for me -------
        Forward pass for batch normalization.
      
        During training the sample mean and (uncorrected) sample variance are
        computed from minibatch statistics and used to normalize the incoming data.
        During training we also keep an exponentially decaying running mean of the
        mean and variance of each feature, and these averages are used to normalize
        data at test-time.
      
        At each timestep we update the running averages for mean and variance using
        an exponential decay based on the momentum parameter:
        
        ------------- WHY ARE WE NOT IMPLEMENTING AT TEST? -----------------
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
      
        Note that the batch normalization paper suggests a different test-time
        behavior: they compute sample mean and variance for each feature using a
        large number of training images rather than using a running average. For
        this implementation we have chosen to use running averages instead since
        they do not require an additional estimation step; the torch7
        implementation of batch normalization also uses running averages.
      
        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean of features
          - running_var Array of shape (D,) giving running variance of features
      
        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        # check for correct dims: img should nt be raveled!
        if input.shape[1] != self.n_neurons:
            raise ValueError('expected probably 4D input (got {}D input) [expected 2D coz for passing the unittest!]'
                             .format(input.dim()))
        # print(input.shape)
        x = input

        sample_mean = x.mean(dim=0)
        sample_var = x.var(dim=0, unbiased=False)
        normalized_x = (x - sample_mean) / torch.sqrt(self.eps + sample_var)

        out = self.gamma * normalized_x + self.beta

        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
    """
      This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
      Using torch.autograd.Function allows you to write a custom backward function.
      The function will be called from the nn.Module CustomBatchNormManualModule
      Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
      pass is done via the backward method.
      The forward pass is not called directly but via the apply() method. This makes sure that the context objects
      are dealt with correctly.

      Example:
        my_bn_fct = CustomBatchNormManualFunction()
        normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization
    
        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: batch-normalized tensor

        TODO:
          Implement the forward pass of batch normalization
          Store constant non-tensor objects via ctx.constant=myconstant
          Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
          Intermediate results can be decided to be either recomputed in the backward pass or to be stored
          for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """
        mu = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)

        var_eps = var + eps
        sqrt_var_eps = var_eps.sqrt()
        x_norm = input - mu
        x_hat = x_norm / sqrt_var_eps

        out = gamma * x_hat + beta

        ctx.save_for_backward(gamma, beta, mu, sqrt_var_eps, x_hat)
        ctx.epsilon = eps

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the batch normalization.

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments

        TODO:
          Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
          Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
          inputs to None. This should be decided dynamically.
        """

        gamma, beta, mu, sqrt_var_eps, x_hat = ctx.saved_tensors

        dL_dgamma = None  # or as smht else?
        # wrt Gamma (we need gamma 2 as well lol)
        if ctx.needs_input_grad[0] | ctx.needs_input_grad[1]:
            dL_dgamma = torch.sum(torch.mul(grad_output, x_hat), dim=0)

        # wrt Beta
        dL_dbeta = None
        if ctx.needs_input_grad[2]:
            dL_dbeta = grad_output.sum(dim=0)

        # B and C 
        B, C = grad_output.shape

        # wrt X
        dL_dx = None
        if ctx.needs_input_grad[0]:
            # for dout
            dL_dy = grad_output
            sum_1 = B * dL_dy
            # for beta, sum over batches
            sum_2 = torch.sum(dL_dy, dim=0)
            # for gamma, sum over batches
            sum_3 = x_hat * torch.sum(x_hat * dL_dy, dim=0)
            # finally
            dL_dx = torch.div(gamma, (B * sqrt_var_eps)) * (sum_1 - sum_2 - sum_3)

        return dL_dx, dL_dgamma, dL_dbeta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        # super(CustomBatchNormManualModule, self).__init__()
        super(CustomBatchNormManualModule, self).__init__()

        # save parameters
        self.n_neurons = n_neurons
        self.eps = eps

        # Initialize gamma and beta
        self.gamma = nn.Parameter(torch.ones(self.n_neurons, dtype=torch.float))
        self.beta = nn.Parameter(torch.zeros(self.n_neurons, dtype=torch.float))

    def forward(self, input):
        """
        Compute the batch normalization via CustomBatchNormManualFunction
    
        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor

        TODO:
          Check for the correctness of the shape of the input tensor.
          Instantiate a CustomBatchNormManualFunction.
          Call it via its .apply() method.
        """
        # check if size matches.
        if input.shape[1] != self.n_neurons:
            raise Exception(f"Size DOES matter! Received {input.shape}, expected {self.n_neurons}")

        batch_normalization = CustomBatchNormManualFunction()
        out = batch_normalization.apply(input, self.gamma, self.beta, self.eps)

        return out
