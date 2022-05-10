from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
from pathlib import Path
from base64 import b64encode
import torch
from torch.utils.data import DataLoader
from torch import nn
import random 
import numpy as np
from .utils import *
import matplotlib.pyplot as plt
from main.resources.export import *
import torch
from main.resources.export import Generator

base = Path("main/resources")
device = 'cpu'
z_dim = 128
gen = torch.load(base/'gen.pt', map_location=torch.device('cpu')).to(device)
dataset = DataLoader(
        MovingMNIST(base/'mmnist.npy', transform=transform), 
        batch_size=1, 
        shuffle=True
    ) 

# Create your views here.
def main(request):
    print('asd',os.getcwd())
    with open(base/"temp.txt", "r") as f:
        data = f.read()
    return HttpResponse(data)

def frontTest(request):
    i = 0
    count = 10
    for real in dataset:
        i+=1
        cur_batch_size = len(real)
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        real = real.to(device)
        
        real_output = real[:,9,:,:].view([cur_batch_size,1,1,64,64])
        real_input = real[:,:9,:,:].view([cur_batch_size,1,9,64,64])
        
        result = gen(fake_noise,real_input).view([-1,64,64]).detach()
        act = real_output.detach().view([-1,64,64])
        
        real_input = real[:,:9,:,:].detach()
        prev = real[:,:9,:,:].detach()
        
        starting_data = torch.cat((prev, act.view([-1,1,64,64])), dim=1)
        
        break

    #  Furture Progression
    frames = []
    future = 3
    for i in range(future):
        new_inp = torch.cat((prev[:,1:9,:,:], act.view([-1,1,64,64])), dim=1)
        result = gen(fake_noise,new_inp).view([-1,64,64]).detach()
        frames.append(result)
        prev = new_inp.detach().clone()
        act = result.detach().clone()
    ####
    
    real_output  = real_output.detach().cpu().numpy()[0,0,0]
    real_output = getSrcFromNumpy(real_output)

    frames = [getSrcFromNumpy(frame[0]) for frame in frames]
    
    real_input = real_input.detach().cpu().numpy()[0]
    real_input = [getSrcFromNumpy(real_input[i]) for i in range(9)]
    data = {'data':{'real_output':real_output, 'real_input':real_input, 'frames':frames}}
    # return render(request, 'main/frontTest.html', data)
    return JsonResponse(data)
