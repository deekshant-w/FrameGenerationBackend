from torch import nn
from main.resources.convlstm import *
import torch

badListVar = None
class PrintLayer(nn.Module):
    def __init__(self, num):
        print(f"DEBUG: {num}")
        self.num = num
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        global badListVar
        # Do your print / debug stuff here
        try:
            print(f"<{self.num}> : {x.shape}")
        except Exception as e:
            print(e)
            print(type(x))
            print(len(x))
            badListVar = x
            raise e
        return x
    
# LSTM() returns tuple of (tensor, (recurrent state))
class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (layer_output_list, last_state_list)
        layer_output_list, _ = x
        # Reshape shape (batch, hidden)
        return layer_output_list[0][:, -1, :, :, :]
    
    
def get_noise_block(in_channel, out_channel, kernel=3, stride=2, last=False):
    if last:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride),
            nn.Tanh()
        )
       
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channel, kernel, stride),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


def get_main_gen_block(input_dim, hidden_dim, kernel_size, num_layers, last=False):
    if last:
        return nn.Sequential(
            ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, return_all_layers=False),
            nn.Tanh()
        )

    return nn.Sequential(
        ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, return_all_layers=False),
        extract_tensor(),
        nn.Tanh(),
        # PrintLayer("Convlstm"),
        # nn.BatchNorm3d(out_channel),
        # nn.LeakyReLU(negative_slope=0.05, inplace=True)
    )


class Generator(nn.Module):
    def __init__(self, batch_size, image_channels=1, z_dim=128, channels=16):
        super(Generator, self).__init__()
        self.z_dim=z_dim
        self.batch_size = batch_size
        self.noiseModel = nn.Sequential(
           # PrintLayer('noise 1'),
           get_noise_block(z_dim, channels*8, stride=1),
           # PrintLayer('noise 2'),
           get_noise_block(channels*8, channels*4, stride=1),
           # PrintLayer('noise 3'),
           get_noise_block(channels*4, channels*4),
           # PrintLayer('noise 4'),
           get_noise_block(channels*4, channels*2, kernel=4, stride=1),
           # PrintLayer('noise 5'),
           get_noise_block(channels*2, channels*2),
           # PrintLayer('noise 6'),
           get_noise_block(channels*2, channels, kernel=4, stride=1),
           # PrintLayer('noise 7'),
           get_noise_block(channels, channels, kernel=4),
           # PrintLayer('noise 8'),
           get_noise_block(channels, image_channels, kernel=3, stride=1, last=True),
           # PrintLayer('noise 9'),
        )

        self.mainModel = nn.Sequential(
           # PrintLayer('gen 1'), 
           get_main_gen_block(1, [32, 64, 128, 1], (3, 3), 4),
           
           # PrintLayer('gen 2'),
           # get_main_gen_block(channels*4, channels*4),
           # PrintLayer('gen 3'),
           # get_main_gen_block(channels*4, channels*2),
           # PrintLayer('gen 4'),
           # get_main_gen_block(channels*2, channels*2, kernel=(3,1,1)),
           # PrintLayer('gen 5'),
           # get_main_gen_block(channels*2, channels, kernel=(3,1,1)),
           # PrintLayer('gen 6'),
           # get_main_gen_block(channels, image_channels, kernel=(2,1,1), last=True),
           # PrintLayer('gen 7'),
        )
        
    def noiseMod(self,noise):
        return noise.view(len(noise), self.z_dim, 1, 1)
        
    def forward(self, noise, prev):
        # print(prev.max())
        # prev -> (B,19,64,64)
        lastFrame = self.noiseModel(self.noiseMod(noise))
        # prev = prev.to('cpu')
        # lastFrame = lastFrame.to('cpu')
        # print(prev.shape, lastFrame.shape)
        # B C T H W (B, 1, 20, 64, 64)
        data = torch.cat((prev.view([self.batch_size,1,9,64,64]), lastFrame.view([self.batch_size,1,1,64,64])), dim=2)
        data = data.view([self.batch_size, 10, 1, 64, 64])
        # print(data.shape)
        res = self.mainModel(data)
        res = res.view([self.batch_size, 1, 1, 64, 64])
        return res


def get_critic_block(input_dim, hidden_dim, kernel_size, num_layers, last=False):
    if last:
        return nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel, stride)
        )
       
    return nn.Sequential(
        ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, return_all_layers=False),
        extract_tensor(),
        # PrintLayer("Criticcc"),
    )

class Critic(nn.Module):
    def __init__(self, batch_size, image_channels=1, channels=16):
        super().__init__()
        self.batch_size=batch_size
        self.dis = nn.Sequential(
            # PrintLayer(1),
            get_critic_block(1, [32, 1], (3, 3), 2),
            nn.Flatten(),
            nn.Linear(64*64, 1),
           # nn.MaxPool3d((1,2,2),(1,2,2)),
           # PrintLayer(2),
           # get_critic_block(image_channels, channels*8),
           # PrintLayer(3),
           # get_critic_block(channels*8, channels*4),
           # PrintLayer(4),
           # get_critic_block(channels*4, channels*4, stride=(1,2,2)),
           # PrintLayer(5),
           # get_critic_block(channels*4, channels*2),
           # PrintLayer(6),
           # get_critic_block(channels*2, channels*2),
           # PrintLayer(7),
           # get_critic_block(channels*2, channels, kernel=4),
           # PrintLayer(8),
           # get_critic_block(channels, channels, kernel=4),
           # PrintLayer(9),
           # get_critic_block(channels, 1, kernel=(4,3,3), last=True),
           # PrintLayer(10)
        )
        
    def forward(self, frame, prev):
        # frame = frame.to('cpu')
        # prev = prev.to('cpu')
        # print(frame.shape)
        # print(prev.shape)
        frame_u = frame.view([self.batch_size,1,1,64,64])
        # print(frame.shape)
        # print(prev.shape)
        prev_u = prev.view([self.batch_size,1,9,64,64])
        data = torch.cat((prev_u,frame_u),dim=2)
        data = data.view([self.batch_size, 10, 1, 64, 64])
        return self.dis(data).view([-1,1])

def get_noise(n_samples, size, device='cuda'):
    return torch.randn(n_samples, size, device=device)