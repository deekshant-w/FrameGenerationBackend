from django.shortcuts import render
from django.http import HttpResponse
import os
from pathlib import Path
from base64 import b64encode

base = Path("main/resources")

# Create your views here.
def main(request):
    print('asd',os.getcwd())
    with open(base/"temp.txt", "r") as f:
        data = f.read()
    return HttpResponse(data)

def frontTest(request):
    with open(base/"giff.gif", "rb") as f:
        data = b64encode(f.read())
    return render(request, 'main/frontTest.html', {'data': data.decode()})