import torch
import torch.nn as nn
import time
import argparse
import os
import datetime

from torch.distributions.categorical import Categorical

# visualization
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import math
import numpy as np
import torch.nn.functional as F
import random
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
####### my own import file ##########
from listofpathpoint import input_handler
import cnc_input
import hybrid_models
####### my own import file ##########
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
'''
so, the models we have are TransEncoderNet,
                            Attention
                            LSTM
                            HPN
each one have initial parameters and the forward part, 
once we have the forward part, the back propagation will 
finished automatically by pytorch  
'''
TOL = 1e-3
TINY = 1e-15
learning_rate = 1e-4   #learning rate
B = 512             #batch size
B_val = 1000        #validation Batchsize
B_valLoop = 20
steps = 2500
n_epoch = 100       # epochs

print('======================')
print('prepare to train')
print('======================')
print('Hyper parameters:')
print('learning rate', learn_rate)
print('batch size', B)
print('validation size', B_val)
print('steps', steps)
print('epoch', n_epoch)
print('======================')

'''
instantiate a training network and a baseline network
'''
temp = input_handler('mother_board.json')
X_val = temp.package_points()
X_val = torch.floatTensor(X_val)
print(len(X_val))
'''
X_val consisted by 'list of list of list'
'rectangle list' 'channel list' 'point xy list' respectively
'''
try:
    del Actor  # remove existing model
    del Critic # remove existing model
except:
    pass
Actor = HPN(n_feature = 2, n_hidden = 128)
Critic = HPN(n_feature = 2, n_hidden = 128)
optimizer = optim.adam(Actor.parameters(), lr=learning_rate)

# Putting Critic model on the eval mode
Actor = Actor.to(device)
Critic = Critic.to(device)
Critic.eval()

epoch_ckpt = 0
tot_time_ckpt = 0

val_mean = []
val_std = []

plot_performance_train = []
plot_performence_baseline = []
# recording the result of the resent epoch makes it available for future
#*********************# Uncomment these lines to load the previous check point
"""
checkpoint_file = "filename of the .pkl"
checkpoint = torch.load(checkpoint_file, map_location=device)
epoch_ckpt = checkpoint['epoch'] + 1
tot_time_ckpt = checkpoint['tot_time']
plot_performance_train = checkpoint['plot_performance_train']
plot_performance_baseline = checkpoint['plot_performance_baseline']
Critic.load_state_dict(checkpoint['model_baseline'])
Actor.load_state_dict(checkpoint['model_train'])
optimizer.load_state_dict(checkpoint['optimizer'])

print('Re-start training with saved checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))

"""
#***********************# Uncomment these lines to load the previous check point

# Main training loop
# The core training concept mainly upon Sampling from the actor
# then taking the greedy action from the critic

start_training_time = time.time()
time_stamp = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S") # Load the time stamp

C = 0       # baseline => the object which the actor can compare
R = 0       # reward

zero_to_bsz = torch.arrange(B, device = device) # a list contains 0 to (batch size -1)

for epoch in range(0, n_epoch):
    # re-start training with saved checkpoint
    epoch += epoch_ckpt # adding the number of the former epochs

    # Train the model for one epoch

    start = time.time() # record the starting time
    Actor.train() #start training actor

    for i in range(1, steps+1): # 1 ~ 2500 steps
        X = X_val
        mask = torch.zero(B,len(X)).cuda() # use mask to make some points impossible to choose
        R= 0
        logprobs = 0
        reward = 0
        Y = X.view(B,len(X))
        x = Y[:,0] #set the single batch to the x 
        h = None
        c = None
        context = None
        Transcontext = None

        # Actor Sampling phase
        for k in range(len(X)):
            context, Transcontext, output, h, c, _ = Actor(context,Transcontext,x=x, X_all=X, h=h, c=c, mask=mask)
            sampler = torch.distributions.Categorical(output)
            idx = sampler.sample()
            # prepare for the back propagation of pytorch
            Y1 = Y[zero_to_bsz, idx.data].clone()
            if k == 0: #if this is the first point 
                Y_ini = Y1.clone()
            if k > 0: 
                reward = torch.sum((Y1-Y0)**2,dim = ?