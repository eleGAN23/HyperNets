from numpy.core.arrayprint import _leading_trailing
import torch
import time
import torch.nn as nn
import wandb
import numpy as np
from matplotlib import pyplot
import math
import sys
from hypercomplex_layers import PHConv



class Trainer():
    def __init__(self, net, optimizer, scheduler, epochs, quat_data, n, 
                      use_cuda=True, gpu_num=0, print_every=100,
                      checkpoint_folder="./checkpoints",
                      saveModelsPerEpoch=True, get_iter_time=False,
                      get_inf_time=False,
                      optim_name="SGD",
                      lr=0.1,
                      momentum=0.9,
                      weight_decay=5e-4,
                      eval_percentage=0.20,
                      l1_reg=False,
                      project_name="phm-resnet-noaug"):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoints_folder = checkpoint_folder
        self.eval_percentage = eval_percentage
        self.get_iter_time = get_iter_time
        self.get_inf_time = get_inf_time
        self.optim_name = optim_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.l1_reg = l1_reg

        if self.use_cuda:
            print("Running on GPU?", self.use_cuda)
            self.net = net.cuda('cuda:%i' %self.gpu_num)
        else:
            self.net = net


    def train(self, train_loader, eval_loader, test_loader):
        # print(self.net)

        criterion = nn.CrossEntropyLoss()


        for epoch in range(self.epochs):  # loop over the dataset multiple times

            start = time.time()
            running_loss_train = 0.0
            running_loss_eval = 0.0
            total = 0.0
            correct = 0.0
            iter_time = []
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' %self.gpu_num), labels.cuda('cuda:%i' %self.gpu_num)

                self.optimizer.zero_grad()

                if self.get_iter_time:
                    start_iter_time = time.time()
                outputs = self.net(inputs)

                loss = criterion(outputs, labels)

                if self.l1_reg:
                    # Add L1 regularization to A
                    regularization_loss = 0.0
                    for child in self.net.children():
                        for layer in child.modules():
                            if isinstance(layer, PHConv):
                                for param in layer.a:
                                    regularization_loss += torch.sum(abs(param))
                    loss += 0.001 * regularization_loss


                loss.backward()

                self.optimizer.step()
         
                
                if self.get_iter_time:
                    end_iter_time = time.time()
                    computed_time = end_iter_time-start_iter_time
                    print("[Time per iter: %f]" %computed_time)
                    iter_time.append(computed_time)

                # print statistics
                running_loss_train += loss.item()
            end = time.time()
            
            if self.get_iter_time:
                print("[Avg time per iter and std]")
                print(np.mean(np.asarray(iter_time)))
                print(np.std(np.asarray(iter_time)))
                break
            for j, eval_data in enumerate(eval_loader, 0):
                inputs, labels = eval_data
                self.net.eval()
                
                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' %self.gpu_num), labels.cuda('cuda:%i' %self.gpu_num)
                eval_outputs = self.net(inputs)
                eval_loss = criterion(eval_outputs, labels)
                running_loss_eval += eval_loss.item()

                _, predicted = torch.max(eval_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100*correct/total

            
            self.scheduler.step()

            print("[Epoch: %i][Train Loss: %f][Val Loss: %f][Val Acc: %f][Time: %f]" %(epoch+1, running_loss_train/i, running_loss_train/j, acc, end-start))
            wandb.log({"train loss": running_loss_train/i})
            wandb.log({"val loss": running_loss_eval/j})
            wandb.log({"val acc": acc})
            
            running_loss_train = 0.0
            running_loss_eval = 0.0
            self.net.train()


        print('Finished Training')
        checkpoint_path = self.checkpoints_folder + "/" + self.net.__class__.__name__ + ".pt"

        # Save checkpoint
        torch.save(self.net.state_dict(), checkpoint_path)


    def test(self, test_loader, get_params=False):
        print("Testing net...")
        self.net.eval()
        checkpoint_path = self.checkpoints_folder + "/" + self.net.__class__.__name__ + ".pt"

        if self.use_cuda:
            self.net = self.net.cuda('cuda:%i' %self.gpu_num)

        self.net.load_state_dict(torch.load(checkpoint_path))

        if get_params:
            print(self.net)
            print("Print first convolution")
            print("MIN:", torch.min(self.net.conv1.a))
            print("MAX:", torch.max(self.net.conv1.a))
            print(sum(self.net.conv1.a))
            print("Mid layer")
            print("MIN:", torch.min(self.net.layer3[0].conv1.a))
            print("MAX:", torch.max(self.net.layer3[0].conv1.a))
            print(sum(self.net.layer3[0].conv1.a))
            print("Last layer")
            print("MIN:", torch.min(self.net.layer4[1].conv2.a))
            print("MAX:", torch.max(self.net.layer4[1].conv2.a))
            print(sum(self.net.layer4[1].conv2.a))



        else:
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    if self.use_cuda:
                        images, labels = images.cuda('cuda:%i' %self.gpu_num), labels.cuda('cuda:%i' %self.gpu_num)
                    if self.get_inf_time:
                        start_inf_time = time.time()
                    # calculate outputs by running images through the network
                    outputs = self.net(images)
                    
                    if self.get_inf_time:
                        end_inf_time = time.time()
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            if self.get_inf_time:
                inference_time = end_inf_time - start_inf_time
                print("Inference time: %f" %inference_time)
            print('Accuracy %s on the test images: %f %%' % (self.net.__class__.__name__,
                100 * correct / total))
            wandb.log({"Test Accuracy": 100*correct/total})
