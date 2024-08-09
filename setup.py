# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name='matrix',
    ext_modules=[
        CUDAExtension(
            name='matrix',
            sources=['extension.cpp', 'matrix_multiplication.cu', 'matrix_add.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    __version__ = "0.1.0"
)