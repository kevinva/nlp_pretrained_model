import numpy as np
import tsensor
import torch

n = 200
d = 764
n_neurons = 100

# W = np.random.randn(d, n_neurons)
# b = np.random.randn(n_neurons, 1)
# X = np.random.randn(n, d)

# with tsensor.clarify():
#     Y = np.matmul(X, W) + b.T

#########################################

# L = torch.nn.Linear(d, n_neurons)
# X = torch.rand(n, n)

# with tsensor.clarify():
#     Y = L(X)

###########################################


# W = np.random.randn(n_neurons, d)
# b = np.random.randn(n_neurons, 1)
# X = np.random.randn(n, d)

# with tsensor.explain():
#     Y = np.matmul(W, X.T) + b


###########################################

batch_size = 10
n_batches = n // batch_size

W = torch.rand(n_neurons, d)
b = torch.rand(n_neurons, 1)
X = torch.rand(n_batches, batch_size, d)

with tsensor.explain():
    for i in range(n_batches):
        batch = X[i,:,:]
        Y = torch.relu(W @ batch.T + b)