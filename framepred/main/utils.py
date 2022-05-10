import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
from base64 import b64encode
import matplotlib.pyplot as plt


class MovingMNIST(Dataset):
    def __init__(self, root_dir, transform=[]):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = np.load(self.root_dir)[:,8000:,...]
        if self.transform:
            self.dataset = self.transform(self.dataset)

    def __len__(self):
        return self.dataset.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        res = self.dataset[:,idx,:,:]
        return res


class CustomNorm(object):
    def __call__(self, data):
        c = 255/2
        temp = data.astype('float32')
        temp -= c
        temp /= c
        return torch.from_numpy(temp)

transform = transforms.Compose([
    CustomNorm(),
])

def getSrcFromNumpy(image):
    # image 64x64
    image = np.uint8(((image + 1) / 2) * 255)
    imageFile = io.BytesIO()
    image = Image.fromarray(image, 'L')
    image = image.convert('RGB')
    image.save(imageFile, format="png")
    imageFile.seek(0)
    image = b64encode(imageFile.read()).decode()
    src = "data:image/png;base64,"+image
    return src
