from django.shortcuts import render
from django.http import HttpResponse
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
from main.resources.export import Generator

base = Path("main/resources")
device = 'cpu'

gen = Generator(1)
gen = torch.load(base/'gen.pt').to(device)

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
    for i in dataset:
        break
    image = i.numpy()
    src = getSrcFromNumpy(image[0,0])
    # with open(base/"giff.gif", "rb") as f:
    #     data = b64encode(f.read())
    return render(request, 'main/frontTest.html', {'data': src})



"""
{
    'inputs':[]9,
    'outputs':[]3,
    'gif':'',
}
"""
    