import numpy as np
import torch
import torch.nn as nn




input_array = np.random.randn(3,4)
zero_array = np.zeros( (3, 1))

print(input_array)

print(zero_array)

print(input_array + zero_array)


arr = np.array([[1,2], [3,4]])
print(len(arr))

input = torch.tensor([[1,2,3],[4,5,6]])
print(input.shape)

token_embedding =  nn.Embedding(10,10)
print(token_embedding(input).shape)