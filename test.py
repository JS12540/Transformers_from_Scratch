import numpy as np

# Define sample data
num_hiddens, num_heads = 100, 5
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = np.array([3, 2])
X = np.random.rand(batch_size, num_queries, num_hiddens)  # Use random data to simulate input queries
Y = np.random.rand(batch_size, num_kvpairs, num_hiddens)  # Use random data to simulate key-value pairs

print(X.shape, Y.shape)