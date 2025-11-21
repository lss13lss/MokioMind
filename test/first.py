import torch
import torch.nn as nn


# x = torch.tensor([1, 2, 3, 4, 5])
# y = torch.tensor([10, 20, 30, 40, 50])

# result = torch.where(x > 3, x, y)
# print(result)  # Output: tensor([10, 20, 30,  4,  5])


# t = torch.arange(0, 10, 2)
# print(t)  # Output: tensor([0, 2, 4, 6


#outer
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])     
# result = torch.outer(a, b)
# print(result)

#cat

# t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# t2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
# result = torch.cat((t1, t2), dim=0)
# print(result)  

# result = torch.cat((t1, t2), dim=1)
# print(result)

# unsqueeze

# t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(t1)
# print(t1.unsqueeze(0))
# print(t1.unsqueeze(1).size())

# dropout

# dropout = nn.Dropout(p=0.5)
# t1 = torch.tensor([1, 2, 3])
# t2 = dropout(t1.float())
# print(t2)

# linear

# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.tensor([1, 3, 5])
# t2 = torch.tensor([[1, 3, 5]])
# output = layer(t2.float())
# print(output)


# #triu
# x = torch.arange(1, 10).view(3, 3)
# print(torch.triu(x, diagonal=0))  # Upper triangular part of the matrix
# print(torch.triu(x, diagonal=1))  # Upper triangular part above the main diagonal