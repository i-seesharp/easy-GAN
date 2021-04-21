import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import pickle

#Getting the celebA dataset
transform = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder("drive/MyDrive/celeb_data",
                                           transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=128, 
                                          shuffle=True,
                                          num_workers=0)

data_loader

images,_ = next(iter(data_loader))

images.shape

img = torch.transpose(images[0], 0,1)
img = torch.transpose(img,1,2)
plt.imshow(img)

class Discriminator(nn.Module):
  def __init__(self,num_channels,activation_slope=0.2):
        super(Discriminator,self).__init__()
        self.slope = activation_slope
        self.num_channels = num_channels
        self.conv1=nn.Conv2d(in_channels=3,
                             out_channels=num_channels,
                             kernel_size=3,
                             stride=2,
                             padding=1)
        self.conv2=nn.Conv2d(in_channels=num_channels,
                             out_channels=num_channels*2,
                             kernel_size=3,
                             stride=2,
                             padding=1)
        self.conv3=nn.Conv2d(in_channels=num_channels*2,
                             out_channels=num_channels*4,
                             kernel_size=3,
                             stride=2,
                             padding=1)
        self.conv4=nn.Conv2d(in_channels=num_channels*4,
                             out_channels=num_channels*8,
                             kernel_size=3,
                             stride=2,
                             padding=1)
        self.conv5=nn.Conv2d(in_channels=num_channels*8,
                             out_channels=num_channels*16,
                             kernel_size=3,
                             stride=2,
                             padding=1)

        self.dense = nn.Linear(num_channels*16*2*2,1)


  def forward(self,x):

      out=F.leaky_relu(self.conv1(x), self.slope)
      out=F.leaky_relu(self.conv2(out), self.slope)
      out=F.leaky_relu(self.conv3(out), self.slope)
      out=F.leaky_relu(self.conv4(out), self.slope)
      out=F.leaky_relu(self.conv5(out), self.slope) 
      out = out.view(-1, self.num_channels*16*2*2)
      out = self.dense(out)
      return out

#Define Discrimnator and test on random input
D = Discriminator(num_channels=32)
sample_output = D(images)
sample_output.shape

class Generator(nn.Module):
  def __init__(self, num_channels, latent_dim):
    super(Generator, self).__init__()
    self.num_channels = num_channels
    self.latent_dim = latent_dim

    self.up1 = nn.ConvTranspose2d(in_channels=self.num_channels*16,
                                  out_channels=num_channels*8,
                                  stride=2,
                                  padding=1,
                                  kernel_size=4)
    self.batch1 = nn.BatchNorm2d(num_channels*8)
    self.up2 = nn.ConvTranspose2d(in_channels=self.num_channels*8,
                                  out_channels=num_channels*4,
                                  stride=2,
                                  padding=1,
                                  kernel_size=4)
    self.batch2 = nn.BatchNorm2d(num_channels*4)
    self.up3 = nn.ConvTranspose2d(in_channels=num_channels*4,
                                  out_channels=num_channels*2,
                                  stride=2,
                                  kernel_size=4,
                                  padding=1)
    self.batch3 = nn.BatchNorm2d(num_channels*2)
    self.up4 = nn.ConvTranspose2d(in_channels=num_channels*2,
                                  out_channels=num_channels,
                                  stride=2,
                                  padding=1,
                                  kernel_size=4)
    self.batch4 = nn.BatchNorm2d(num_channels)
    self.up5 = nn.ConvTranspose2d(in_channels=num_channels,
                                  out_channels=3,
                                  stride=2,
                                  padding=1,
                                  kernel_size=4)
    
    self.dense = nn.Linear(self.latent_dim, self.num_channels*16*2*2)
  
  def forward(self,x):
    out = self.dense(x)
    out = out.view(-1, self.num_channels*16,2,2)
    out = F.relu(self.batch1(self.up1(out)))
    out = F.relu(self.batch2(self.up2(out)))
    out = F.relu(self.batch3(self.up3(out)))
    out = F.relu(self.batch4(self.up4(out)))
    out = torch.tanh(self.up5(out))

    return out

# Defining generator and trying random input
G = Generator(num_channels=32,latent_dim=256)
latent_code = torch.randn((32,256))
generated_img = G(latent_code)
generated_img.shape

#If cuda is available then, use CUDA
if torch.cuda.is_available():
  print("CUDA Availaible")
  D.cuda()
  G.cuda()

#Hyperparameters
adam_beta = (0.5,0.999)
adam_lr = 2e-4

def D_loss(real,fake,scale=0.95):
    real_batch_size = real.size(0)
    fake_batch_size = fake.size(0)
        
    ones = torch.ones(real_batch_size) * scale
    zeros = torch.zeros(fake_batch_size)

    if torch.cuda.is_available():
      ones = ones.cuda()
      zeros = zeros.cuda()
    criterion = nn.BCEWithLogitsLoss()
    real_loss = criterion(real.squeeze(),ones)
    fake_loss = criterion(fake.squeeze(),zeros)
    return real_loss + fake_loss

def G_loss(fake):
    batch_size = fake.size(0)
    ones = torch.ones(batch_size)

    if torch.cuda.is_available():
      ones = ones.cuda()
    
    criterion=nn.BCEWithLogitsLoss()
    loss = criterion(fake.squeeze(),ones)
    return loss    
  
#Defining Adam Optimizers

d_optimizer = torch.optim.Adam(D.parameters(),lr=adam_lr,betas=adam_beta)
g_optimizer = torch.optim.Adam(G.parameters(),lr=adam_lr,betas=adam_beta)

def train(num_epochs):
    
    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    samples = []
    losses = []

    sample_size = 9
    sample_noise = torch.randn((sample_size, 256)).float()
  
    if torch.cuda.is_available():
        sample_noise = sample_noise.cuda()

    for epoch in range(num_epochs):

        for batch_i, (images, _) in enumerate(data_loader):

            batch_size = images.size(0)
            images = images*2.0 - 1.0

            d_optimizer.zero_grad()
            
            if torch.cuda.is_available():
                images = images.cuda()
                
            D_real = D(images)

            z = torch.randn((batch_size, 256)).float()
            if torch.cuda.is_available():
                z = z.cuda()

            D_fake = D(G(z))
 
            d_loss = D_loss(D_real, D_fake)
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()

            z = torch.randn((batch_size, 256)).float()
            if torch.cuda.is_available():
                z = z.cuda()
            
            D_fake = D(G(z))
            g_loss = G_loss(D_fake)

            g_loss.backward()
            g_optimizer.step()

            if batch_i % 250 == 0:

                curr_stats = {"d_loss": d_loss.item(), "g_loss": g_loss.item()}
                losses.append(curr_stats)
                print('Epoch : {} d_loss: {} g_loss: {}'.format(
                        epoch, d_loss.item(), g_loss.item()))


        G.eval() 
        sample_imgs = G(sample_noise)
        samples.append(sample_imgs)
        G.train()

        plot_images(sample_imgs) 

    with open('samples.pkl', 'wb') as f:
        pickle.dump(samples, f)
    
    torch.save(D.state_dict(), "discriminator.weights")
    torch.save(G.state_dict(), "generator.weights")

    return losses

def plot_images(images):
  fig, axes = plt.subplots(figsize=(9,9), nrows=3, ncols=3, sharey=True, sharex=True)
  for ax, img in zip(axes.flatten(), images):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    im = ax.imshow(img.reshape((64,64,3)))
  plt.show()

losses = train(num_epochs=50)

#Plot the loss curve
d_losses = [x["d_loss"] for x in losses]
g_losses = [x["g_loss"] for x in losses]

plt.plot(range(len(d_losses)), d_losses, label="Discriminator Losses")
plt.plot(range(len(g_losses)),g_losses,label="Generator Losses")
plt.legend()
plt.show()

