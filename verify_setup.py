#!/usr/bin/env python3
"""Verify FLPoison environment setup. 验证环境是否安装正确"""

import sys

def check_imports():
    """Verify all critical imports."""
    try:
        import torch
        import torchvision
        import numpy as np
        import scipy
        import sklearn
        import matplotlib
        import yaml
        import hdbscan
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_pytorch():
    """Verify PyTorch configuration."""
    import torch
    print(f"\nPyTorch Information:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  MPS Available: {torch.backends.mps.is_available()}")

def check_directories():
    """Verify directory structure."""
    import os
    required = ['fl', 'attackers', 'aggregators', 'datapreprocessor', 'configs']
    optional = ['data', 'logs', 'running_caches']
    
    print(f"\nDirectory Structure:")
    for d in required:
        exists = os.path.isdir(d)
        print(f"  {'✓' if exists else '✗'} {d}/ {'(required)' if not exists else ''}")
    
    for d in optional:
        exists = os.path.isdir(d)
        print(f"  {'✓' if exists else '○'} {d}/ (optional, {'exists' if exists else 'will be created'})")

def check_config_files():
    """Verify configuration files."""
    import os
    import glob
    
    configs = glob.glob('configs/*.yaml')
    print(f"\nConfiguration Files: {len(configs)} found")
    if configs:
        print(f"  Sample: {configs[0]}")

if __name__ == '__main__':
    print("=" * 60)
    print("FLPoison Environment Verification")
    print("=" * 60)
    
    if not check_imports():
        sys.exit(1)
    
    check_pytorch()
    check_directories()
    check_config_files()
    
    print("\n" + "=" * 60)
    print("✓ Environment verification complete!")
    print("=" * 60)