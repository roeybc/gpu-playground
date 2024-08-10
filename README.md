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

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgements
* This project uses [PyTorch](https://pytorch.org/), which is licensed under the [BSD-3-Clause License](https://github.com/pytorch/pytorch/blob/master/LICENSE).

* This project uses [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit), which is subject to the [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html).

To run this project, you acknowledge that you must download and install the CUDA Toolkit from NVIDIA's official website and agree to comply with the terms of the EULA. This project does not include or redistribute any proprietary components of the CUDA Toolkit.