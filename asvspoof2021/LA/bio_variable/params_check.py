import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config yaml file')
    
    
    args = parser.parse_args()
    dir_yaml = args.config

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
    
    