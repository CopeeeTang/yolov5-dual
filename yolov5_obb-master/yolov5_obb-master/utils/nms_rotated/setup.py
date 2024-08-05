#!/usr/bin/env python
import os
from setuptools import setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name, module, sources, sources_cuda=[], sources_cuda_later=[]):
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        if torch.__version__ < '1.11' or len(sources_cuda_later) == 0:
            sources += sources_cuda
        else:
            sources += sources_cuda_later
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


# python setup.py develop
if __name__ == '__main__':
    # write_version_py()
    setup(
        name='nms_rotated',
        ext_modules=[
            make_cuda_ext(
                name='nms_rotated_ext',
                module='',
                sources=[
                    'src/nms_rotated_cpu.cpp',
                    'src/nms_rotated_ext.cpp'
                ],
                sources_cuda=[
                    'src/nms_rotated_cuda.cu',
                    'src/poly_nms_cuda.cu',
                ],
                sources_cuda_later=[
                    'src/nms_rotated_cuda.cu',
                    'src/poly_nms_cuda_1.11.cu',
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)