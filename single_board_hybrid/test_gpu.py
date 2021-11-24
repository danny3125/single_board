import torch
import torch.nn as nn
import time
import argparse
import os
import datetime

from torch.distributions.categorical import Categorical

# visualization 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cpu");
gpu_id = -1  # select CPU

gpu_id = '0'  # select a single GPU
# gpu_id = '2,3' # select multiple GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0), gpu_id))

print(device)