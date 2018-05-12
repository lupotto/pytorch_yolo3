from __future__ import print_function
import sys

'''
if len(sys.argv) != 4:
    print('Usage:')
    print('python train.py datacfg cfgfile weightfile')
    exit()
'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from auteltools import auteldata as ad
import random
import math
import os
from util import *
from cfg import parse_cfg
from darknet import  Darknet


#training settings
#datacfg       = sys.argv[1]
#cfgfile       = sys.argv[2]
#weightfile    = sys.argv[3]
datacfg        = 'cfg/autel.data'
cfgfile        = 'cfg/yolov3.cfg'
weightfile     = 'yolov3.weights'

#loading dicts
data_options   = read_data_cfg(datacfg)
net_options    = parse_cfg(cfgfile)[0]

#parameters
trainlist     = data_options['train']
testlist      = data_options['valid']
backupdir     = data_options['backup']
nsamples      = file_lines(trainlist)
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
max_epochs    = max_batches*batch_size/nsamples+1
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
save_interval = 10  # epoches
dot_interval  = 70  # batches

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5


if not os.path.exists(backupdir):
    os.mkdir(backupdir)

seed          = int(time.time())
use_cuda      = True

#seed for generate random rumbers
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

model       = Darknet(cfgfile)
region_loss = model.loss()