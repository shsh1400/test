import torch
import torch.nn as nn
import numpy as np
# # target output size of 5x7
# m = nn.AdaptiveAvgPool2d((5, 7))
# input = torch.randn(1, 64, 8, 9)
# output = m(input)
# print(output.size())
# # target output size of 7x7 (square)
# m = nn.AdaptiveAvgPool2d(7)
# input = torch.randn(1, 64, 10, 9)
# output = m(input)
# print(output.size())
# # target output size of 10x7
# m = nn.AdaptiveAvgPool2d((None, 7))
# input = torch.randn(1, 64, 10, 9)
# output = m(input)
# print(output.size())
# data = torch.tensor(
#         [[[ [9.0, 0, 7, 6],
#             [3, 2, 6, 8],
#             [7, 5, 4, 4],
#             [4, 8, 3, 5]],
#
#          [  [3, 8, 7, 2],
#             [9, 6, 1, 2],
#             [2, 0, 8, 0],
#             [2, 9, 8, 4]]],
#
#          [[ [6, 1, 5, 6],
#             [2, 3, 4, 8],
#             [5, 3, 3, 3],
#             [4, 1, 8, 4]],
#
#           [ [3, 6, 5, 4],
#             [4, 9, 8, 5],
#             [7, 1, 5, 4],
#             [4, 4, 8, 6]]]
#          ]
# )
# print(data.size())
# mean_data = torch.mean(data, dim=1, keepdim=True)
# print(mean_data.size())
# print(mean_data)

# a = np.ones((5,))
# print(a)
# print(a[[True,True,False,True,False]])

# xs = np.broadcast_to(
#             np.linspace(0, 5 - 1, num=5).reshape(1, 5),
#             (5, 5))
# print(xs)
_batch_input1 = torch.tensor(np.linspace(1, 30000,30000).reshape((3,100,100)))
_batch_input2 = torch.tensor(np.linspace(1,3000000,3000000).reshape((3,1000,1000)))
_batch_input = [_batch_input1, _batch_input2]
# print(_batch_input)
# _batch_input = _batch_input[[2, 1, 0], ...]
# print(_batch_input)
# print(_batch_input.dim())
all_sizes: torch.Tensor = torch.Tensor(
    [tensor.shape for tensor in _batch_input])
max_sizes = torch.ceil(
    torch.max(all_sizes, dim=0)[0] / 32) * 32
print(all_sizes)
print(torch.max(all_sizes, dim=0)[0])

padded_sizes = max_sizes - all_sizes
padded_sizes[:, 0] = 0
print(padded_sizes)

pad = torch.zeros(2, 2 * 3, dtype=torch.int)

print(padded_sizes[:, range(3 - 1, -1, -1)])
pad[:, 1::2] = padded_sizes[:, range(3 - 1, -1, -1)]
print(pad)