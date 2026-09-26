#!/usr/bin/env python3
"""
Tràuna AI — check_gpu.py

Diagnostic script. Run this first to verify the environment.

Usage:
  python check_gpu.py
"""

import sys
import platform


def main():
    print('=' * 60)
    print('  Tràuna AI — Environment Diagnostic')
    print('=' * 60)

    # Python
    print(f'\nPython Version : {sys.version}')
    print(f'Platform       : {platform.platform()}')

    # PyTorch
    try:
        import torch
        print(f'\nPyTorch Version: {torch.__version__}')
    except ImportError:
        print('\n[ERROR] PyTorch not installed. Run: pip install torch')
        sys.exit(1)

    # CUDA
    cuda_available = torch.cuda.is_available()
    print(f'CUDA Available : {cuda_available}')

    if cuda_available:
        print(f'CUDA Version   : {torch.version.cuda}')
        gpu_count = torch.cuda.device_count()
        print(f'GPU Count      : {gpu_count}')

        for i in range(gpu_count):
            name  = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_total = props.total_memory / (1024 ** 3)
            vram_free  = (props.total_memory - torch.cuda.memory_reserved(i)) / (1024 ** 3)

            print(f'\nGPU [{i}]')
            print(f'  Name         : {name}')
            print(f'  VRAM Total   : {vram_total:.2f} GB')
            print(f'  VRAM Free    : {vram_free:.2f} GB')
            print(f'  Compute Cap  : {props.major}.{props.minor}')
    else:
        print('\n[WARNING] CUDA not available. AI generation will run on CPU (very slow).')
        print('  Ensure NVIDIA drivers and CUDA toolkit are installed.')
        print('  Install PyTorch with CUDA: https://pytorch.org/get-started/locally/')

    # Diffusers
    try:
        import diffusers
        print(f'\nDiffusers      : {diffusers.__version__}')
    except ImportError:
        print('\n[WARNING] Diffusers not installed. Run: pip install diffusers')

    # Transformers
    try:
        import transformers
        print(f'Transformers   : {transformers.__version__}')
    except ImportError:
        print('[WARNING] Transformers not installed.')

    # Accelerate
    try:
        import accelerate
        print(f'Accelerate     : {accelerate.__version__}')
    except ImportError:
        print('[WARNING] Accelerate not installed.')

    # Pillow
    try:
        from PIL import Image
        import PIL
        print(f'Pillow         : {PIL.__version__}')
    except ImportError:
        print('[WARNING] Pillow not installed.')

    print('\n' + '=' * 60)
    if cuda_available:
        print('  Environment looks good. Proceed to test_generation.py')
    else:
        print('  CUDA not found. Check driver / CUDA installation before continuing.')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
