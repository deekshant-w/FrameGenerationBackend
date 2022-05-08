from django.shortcuts import render
from django.http import HttpResponse
import os
from pathlib import Path

base = Path("main/resources")

# Create your views here.
def main(request):
    print('asd',os.getcwd())
    with open(base/"temp.txt", "r") as f:
        data = f.read()
    return HttpResponse(data)