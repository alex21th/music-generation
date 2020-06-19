import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ops import *


MODEL_NAME = 'model_emb_advanced'

class sample_generator(nn.Module):
    def __init__(self, pitch_range):
        super(sample_generator, self).__init__()
        self.gf_dim   = 64
        self.y_dim   = 100
        self.n_channel = 256

        self.h1      = nn.ConvTranspose2d(in_channels=256, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h2      = nn.ConvTranspose2d(in_channels=192, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h3      = nn.ConvTranspose2d(in_channels=176, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h4      = nn.ConvTranspose2d(in_channels=160, out_channels=1, kernel_size=(1,pitch_range), stride=(1,2))

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,pitch_range), stride=(1,2))
        self.h1_prev = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(2,1), stride=(2,2))
        self.h2_prev = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(2,1), stride=(2,2))
        self.h3_prev = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,1), stride=(2,2))
        
        self.linear1 = nn.Linear(100,256)

        self.embedding = nn.Embedding(25, 256)

        self.bn0_prev = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True)
        self.bn1_prev = nn.BatchNorm2d(48, eps=1e-05, momentum=0.9, affine=True)
        self.bn2_prev = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.bn3_prev = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        
        self.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)

        self.bn1d_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.9, affine=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, z, prev_x, y ,batch_size,pitch_range):

        y = self.embedding(y)
        y = y.view(batch_size, 16, 16, 1)
        
        h0_prev = self.lrelu(self.bn0_prev(self.h0_prev(prev_x)))
        h0_prev = torch.cat((h0_prev, y), 1)
        h1_prev = self.lrelu(self.bn1_prev(self.h1_prev(h0_prev)))
        h2_prev = self.lrelu(self.bn2_prev(self.h2_prev(h1_prev)))  
        h3_prev = self.lrelu(self.bn3_prev(self.h3_prev(h2_prev)))

        h0 = self.lrelu(self.bn1d_1(self.linear1(z)))    #(72,256)
        
        h1 = h0.view(batch_size, self.gf_dim * 2, 2, 1)     #(72,128,2,1)
        h1 = conv_prev_concat(h1,h3_prev)  #(72, 256, 2, 1) 157->228+16=244

        h2 = self.lrelu(self.bn2(self.h1(h1)))  #(72, 128, 4, 1)
        h2 = conv_prev_concat(h2,h2_prev)  #([72, 192, 4, 1])

        h3 = self.lrelu(self.bn3(self.h2(h2)))  #([72, 128, 8, 1]) 
        h3 = conv_prev_concat(h3,h1_prev) #([72, 176, 8, 1])

        h4 = self.lrelu(self.bn4(self.h3(h3)))  #([72, 128, 16, 1])
        h4 = conv_prev_concat(h4,h0_prev) #([72, 144, 16, 1])

        g_x = self.sigmoid(self.h4(h4)) #([72, 1, 16, 128])

        return g_x



class generator(nn.Module):
    def __init__(self,pitch_range):
        super(generator, self).__init__()
        self.gf_dim   = 64
        self.y_dim   = 100
        self.n_channel = 256

        self.h1      = nn.ConvTranspose2d(in_channels=256, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h2      = nn.ConvTranspose2d(in_channels=192, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h3      = nn.ConvTranspose2d(in_channels=176, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h4      = nn.ConvTranspose2d(in_channels=160, out_channels=1, kernel_size=(1,pitch_range), stride=(1,2))

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,pitch_range), stride=(1,2))
        self.h1_prev = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(2,1), stride=(2,2))
        self.h2_prev = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(2,1), stride=(2,2))
        self.h3_prev = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,1), stride=(2,2))
        
        self.linear1 = nn.Linear(100,256)

        self.embedding = nn.Embedding(25, 256)

        self.bn0_prev = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True)
        self.bn1_prev = nn.BatchNorm2d(48, eps=1e-05, momentum=0.9, affine=True)
        self.bn2_prev = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.bn3_prev = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        
        self.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)

        self.bn1d_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.9, affine=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, z, prev_x, y, batch_size, pitch_range):

        y = self.embedding(y)
        y = y.view(batch_size, 16, 16, 1)
        
        h0_prev = self.lrelu(self.bn0_prev(self.h0_prev(prev_x)))   #[72, 16, 16, 1]
        
        h0_prev = torch.cat((h0_prev, y), 1)
        h1_prev = self.lrelu(self.bn1_prev(self.h1_prev(h0_prev)))  #[72, 48, 8, 1]
        h2_prev = self.lrelu(self.bn2_prev(self.h2_prev(h1_prev)))  #[72, 64, 4, 1]
        h3_prev = self.lrelu(self.bn3_prev(self.h3_prev(h2_prev)))  #[72, 128, 2, 1])

        h0 = self.lrelu(self.bn1d_1(self.linear1(z)))    #(72,256)
        
        h1 = h0.view(batch_size, self.gf_dim * 2, 2, 1)     #(72,128,2,1)
        h1 = conv_prev_concat(h1,h3_prev)  #(72, 256, 2, 1) 157->228+16=244

        h2 = self.lrelu(self.bn2(self.h1(h1)))  #(72, 128, 4, 1)
        h2 = conv_prev_concat(h2,h2_prev)  #([72, 192, 4, 1])

        h3 = self.lrelu(self.bn3(self.h2(h2)))  #([72, 128, 8, 1]) 
        h3 = conv_prev_concat(h3,h1_prev) #([72, 176, 8, 1])

        h4 = self.lrelu(self.bn4(self.h3(h3)))  #([72, 128, 16, 1])
        h4 = conv_prev_concat(h4,h0_prev) #([72, 144, 16, 1])

        g_x = self.sigmoid(self.h4(h4)) #([72, 1, 16, 128])

        return g_x


class discriminator(nn.Module):
    def __init__(self,pitch_range):
        super(discriminator, self).__init__()

        self.df_dim = 64
        self.dfc_dim = 1024
        self.y_dim = 100

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,pitch_range), stride=(1,1))
        self.h1_prev = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,1), stride=(1,1))
        self.h2_prev = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,1), stride=(1,1))
        self.h3_prev = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,1), stride=(1,1))
        
        self.linear1 = nn.Linear(1024,1000)
        self.linear2 = nn.Linear(1100,1)

        self.embedding = nn.Embedding(25,100)

        self.bn0 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)

        self.bn1d = nn.BatchNorm1d(1000, eps=1e-05, momentum=0.9, affine=True)

        self.lrelu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()


    def forward(self,x,y,batch_size,pitch_range):        
        y = self.embedding(y)
        y = y.view(-1, 100)
        
        h0 = self.lrelu(self.bn0(self.h0_prev(x))) # 72, 32, 13, 1
        fm = h0

        h1 = self.lrelu(self.bn1(self.h1_prev(h0)))  #torch.Size([72, 64, 10, 1])

        h2 = self.lrelu(self.bn2(self.h2_prev(h1))) # [72, 128, 7, 1]

        h3 = self.lrelu(self.bn3(self.h3_prev(h2))) # [72, 256, 4, 1]
        h3 = h3.view(batch_size, -1)

        h4 = self.lrelu(self.bn1d(self.linear1(h3)))
        h4 = torch.cat((h4,y),1) # 72x1256
        h5 = self.linear2(h4)

        h5_sigmoid = self.sigmoid(h5)

        return h5_sigmoid, h5, fm