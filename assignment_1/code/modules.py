"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        # TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        # Initialization for the trainable parameters
        mu_w, sigma_w = (0, 0.0001)
        mu_b, sigma_b = (0, 0.0002)

        params_weights_init = np.random.normal(mu_w, sigma_w, (out_features, in_features))
        # need add a dimension so that we add bias to each image in a batch
        params_biases_init = np.random.normal(mu_b, sigma_b, (out_features, 1))

        # Initialization for the gradients so just to create them
        mu_w, sigma_w = (0, 0.0001)
        mu_b, sigma_b = (0, 0.0002)

        grads_weights_init = np.random.normal(mu_w, sigma_w, (in_features, out_features))
        grads_biases_init = np.random.normal(mu_b, sigma_b, (out_features, 1))

        # Set the elements of the network to the initialization defaults
        self.params = {'weight': params_weights_init, 'bias': params_biases_init}
        self.grads = {'weight': grads_weights_init, 'bias': grads_biases_init}

        # Additional state variables needed
        self.input = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # Probably need to do Wx + b
        # transpose so to arrive to dims like [n_images, n_features]
        out = np.transpose(np.matmul(self.params['weight'], x.T) + self.params['bias'])

        # Store so we can use them when backpropagating
        self.input = x

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        # d_L/d_x = d_L/d_a * d_a/d_x
        # d_a/d_x = W
        # d_L/d_a = dout

        # Return dL/dx coz will be used to calculate dL/da
        dx = np.matmul(dout, self.params["weight"])

        # d_L/d_b just the da/dx coz da/b is just identity
        self.grads["bias"] = np.reshape(dout.sum(axis=0), self.grads["bias"].shape)

        # dL/dW
        self.grads["weight"] = np.matmul(dout.T, self.input)

        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # ReLU function
        out = np.maximum(x, 0)

        # Store all the values for which we are going to have a derivative
        self.input_mask = x > 0

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        # Propagate only activated `neurons`
        dx = dout * self.input_mask

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass. Squeee your inputs in the range [0:1]
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # x is multidimensional, thus we need to
        # add michine epsilon to not dived by zero later (maxx - maxx = 0)
        maxx = x.max(axis=1) - np.finfo(float).eps
        num = np.exp(x - maxx.reshape((-1, 1)))
        denom = num.sum(axis=1)
        out = num / denom.reshape((-1, 1))

        self.probs_softmax = out

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        # See the derivation in LaTex. WTFFF is here ?

        # We need to perfrom the pass on all the inputs of the batch
        batch_size = self.probs_softmax.shape[0]
        dim = self.probs_softmax.shape[1]

        diag_xN = np.zeros((batch_size, dim, dim))
        ii = np.arange(dim)
        diag_xN[:, ii, ii] = self.probs_softmax

        dxdx_t = diag_xN - np.einsum('ij, ik -> ijk', self.probs_softmax, self.probs_softmax)

        dx = np.einsum('ij, ijk -> ik', dout, dxdx_t)

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        # Calculate loss (i already handle zero cases, but still add eps)
        out = np.sum(-1 * y * np.log(x + np.finfo(float).eps), axis=1).mean()

        return out

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """
        # for cross-entropy:

        dx = -np.divide(y, x) / len(y)

        return dx


if __name__ == '__main__':
    lin_layer = LinearModule(10, 10)
