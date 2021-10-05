import argparse
import math
import sys
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.optim.lr_scheduler import (CosineAnnealingLR, CyclicLR, MultiStepLR,
                                      StepLR)

sys.path.append('../utils')
import argparse
import os
# from torch import nn
import random
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.optim as optim

from GetModel import GetModel
from training import Trainer
from utils.dataloaders import (CIFAR10_dataloader, CIFAR100_dataloader,
                               STL10_dataloader, SVHN_dataloader)
from utils.readFile import readFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()#fromfile_prefix_chars='@')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1656079)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--n_workers', default=1)
    parser.add_argument('--quat_data', type=bool, default=False)
    parser.add_argument('--n', type=int, default=4, help="n parameter for PHM layer")
    parser.add_argument('--optim', type=str, default="SGD")
    parser.add_argument('--scheduler', type=str, default="cosine")
    parser.add_argument('--l1_reg', type=bool, default=False)
    parser.add_argument('--train_dir', type=str, default='./data/', help="Folder containg training data. It must point to a folder with images in it.")
    
    parser.add_argument('--Dataset', type=str, default='SVHN', help='SVHN, CIFAR10')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='resnet20', help='Models: ...')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', default=(0.0, 0.9))
    parser.add_argument('--print_every', type=int, default=50, help='Print Loss every n iterations')
    parser.add_argument('--get_iter_time', type=bool, default=False)
    parser.add_argument('--get_inf_time', type=bool, default=False)
    parser.add_argument('--EpochCheckpoints', type=bool, default=True, help='Save model every epoch. If set to False the model will be saved only at the end')
    
    parser.add_argument('--TextArgs', type=str, default='TrainingArguments.txt', help='Path to text with training settings')

    parse_list=readFile(parser.parse_args().TextArgs)
    
    opt = parser.parse_args(parse_list)

    use_cuda = opt.cuda
    gpu_num = opt.gpu_num
    
    lr = opt.lr
    # betas = opt.betas.replace(',', ' ').split()
    betas = (float(opt.betas[0]), float(opt.betas[1]))
    epochs = opt.epochs
    img_size = opt.image_size
    batch_size = opt.batch_size
    print_every = opt.print_every
    EpochCheckpoints = opt.EpochCheckpoints
    dataset = opt.Dataset
    n_workers = opt.n_workers
    l1_reg = opt.l1_reg
    
    if n_workers=='max':
        n_workers = cpu_count()
    




    # Set seed    
    manualSeed = opt.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    seed=manualSeed
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model = opt.model
    quat_data = opt.quat_data
    n = opt.n
    get_iter_time = opt.get_iter_time
    get_inf_time = opt.get_inf_time
    train_dir = opt.train_dir

    
    if dataset == 'SVHN':
        train_loader, test_loader, eval_loader, data_name = SVHN_dataloader(root=train_dir, quat_data = quat_data, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
        num_classes=10
        
    elif dataset == 'CIFAR10':
        train_loader, test_loader, eval_loader, data_name = CIFAR10_dataloader(root=train_dir, quat_data = quat_data, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
        num_classes=10

    elif dataset == 'CIFAR100':
        train_loader, test_loader, eval_loader, data_name = CIFAR100_dataloader(root=train_dir, quat_data = quat_data, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
        num_classes = 100

    elif dataset == 'STL10':
        train_loader, test_loader, eval_loader, data_name = STL10_dataloader(root=train_dir, quat_data = quat_data, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
        num_classes=10

    else:
        RuntimeError('Wrong dataset or not implemented')

    
    
    net = GetModel(str_model=model, quat_data=quat_data, n=n, num_classes=num_classes)
    
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters:', params)
    print()
    

    wandb.init(project="phm-resnet")
    config = wandb.config
    wandb.watch(net)
    wandb.config.update(opt)


    
        
    checkpoint_folder = 'checkpoints/'
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    # Initialize optimizers
    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas)
    weight_decay_cifar = 5e-4
    weight_decay_custom = 0.0001 #best
    if opt.optim == "SGD":
        optim_name = "SGD"
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    if opt.optim == "Adam":
        optim_name = "Adam"
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=opt.weight_decay, betas=(0.5, 0.999))
    
    # Add scheduler
    if opt.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=200)
    elif opt.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=50)
    elif opt.scheduler == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[59, 119, 159], gamma=0.15)
    elif opt.scheduler == "CyclicLR":
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)


    '''Train model'''
    trainer = Trainer(net, optimizer, scheduler, epochs=epochs, quat_data=quat_data, n=n, 
                      use_cuda=use_cuda, gpu_num=gpu_num, print_every = print_every,
                      checkpoint_folder = checkpoint_folder,
                      saveModelsPerEpoch=EpochCheckpoints,
                      get_iter_time=get_iter_time,
                      get_inf_time=get_inf_time,
                      optim_name=optim_name,
                      lr=lr,
                      momentum=opt.momentum,
                      weight_decay=opt.weight_decay,
                      l1_reg=l1_reg
                      )
    
    
    trainer.train(train_loader, eval_loader, test_loader)
    trainer.test(test_loader, get_params=False)
    

############################################################################################


############################################################################################
