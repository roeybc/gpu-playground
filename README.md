# GPU Playground
Various cuda kernels, trying the leverage the GPU as efficiently as possible. 
Available in python through pytorch extensions. 

### How to use
Run `run.sh` to build & run sample in python.

### Usage
To run the library in python, see the following: 
```python
import torch
import matrix

a = torch.randn([12,12]).contiguous().float().cuda()
b = torch.randn([12,12]).contiguous().float().cuda()

add = matrix.add(a, b)
```