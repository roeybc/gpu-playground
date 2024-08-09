import torch
import matrix

a = torch.randn([12,12]).contiguous().float().cuda()
b = torch.randn([12,12]).contiguous().float().cuda()

add = matrix.add(a, b)
print(torch.isclose(add, a+b).all())

a = torch.randn([10,13]).contiguous().float().cuda()
b = torch.randn([13,5]).contiguous().float().cuda()
mul = matrix.mul(a, b)
cuda_mul = a@b

print(torch.isclose(mul, cuda_mul).all())