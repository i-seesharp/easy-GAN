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

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import datetime
import os, sys
import glob
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image

#Defining Custom Layers
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super(LeakyReLU, self).__init__(negative_slope=negative_slope, inplace=inplace)

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * torch.rsqrt((x**2).mean(dim=1, keepdim=True) + epsilon)

class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()

    def forward(self, x):
        N, _, H, W = x.size()
        std = torch.std(x, dim=0, keepdim=True)
        std_mean = torch.mean(std, dim=(1,2,3), keepdim=True).expand(N, -1, H, W)
       
        return torch.cat([x, std_mean], dim=1)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, dilation=1, groups=1,
                    lrelu=True, weight_norm=False, pnorm=True, equalized=True):
        super(Conv2d, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2 * dilation
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups
        )
        if weight_norm:
            nn.utils.weight_norm(self.conv)
        
        self.lrelu = LeakyReLU() if lrelu else None
        self.normalize = PixelNorm() if pnorm else None
        self.equalized = equalized
        if equalized:
            self.conv.weight.data.normal_(0, 1)
            fan_in = np.prod(self.conv.weight.size()[1:])
            self.he_constant = np.sqrt(2.0/fan_in)
            self.conv.bias.data.fill_(0.)
        
    def forward(self, x): 
        y = self.conv(x)
        y = y*self.he_constant if self.equalized else y
        y = self.lrelu(y) if self.lrelu is not None else y
        y = self.normalize(y) if self.normalize is not None else y
        return y

class Linear(nn.Module):
    def __init__(self, in_dims, out_dims, 
                weight_norm=False, equalized=True):
        super(Linear, self).__init__()
        
        self.linear = nn.Linear(in_dims, out_dims)
        if weight_norm:
            nn.utils.weight_norm(self.linear)
        self.equalized = equalized
        if equalized:
            self.linear.weight.data.normal_(0, 1)
            fan_in = np.prod(self.linear.weight.size()[1:])
            self.he_constant = np.sqrt(2.0/fan_in)
            self.linear.bias.data.fill_(0.)

    def forward(self, x): 
        y = self.linear(x)
        y = y*self.he_constant if self.equalized else y
        return y

class GANLoss(nn.Module):
    def __init__(self, loss_type='lsgan', device='cuda'):
        super(GANLoss, self).__init__()
        self.loss_type = loss_type
        real_label = torch.tensor(1.0).to(device)
        fake_label = torch.tensor(0.0).to(device)
        if loss_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.loss = nn.BCELoss()
        elif loss_type == 'wgan-gp':
            self.loss = wasserstein_loss
            fake_label = torch.tensor(-1.0).to(device)
        self.register_buffer('real_label', real_label)
        self.register_buffer('fake_label', fake_label)
    
    def __call__(self, inputs, is_real=None):
        if is_real is not None:
            labels = self.real_label.expand_as(inputs) if is_real else self.fake_label.expand_as(inputs)
            loss = self.loss(inputs, labels)
        return loss

def wasserstein_loss(inputs, labels):
    return torch.sum(labels*-inputs)

class PGGAN(nn.Module):
    def __init__(self, args, device='cuda'):
        super().__init__()
        self.args = args
        self.device = device
        self.img_channels = 3
        self.depths = [args.zdim, 256, 256, 256, 64*2, 64*2]
        self.didx = 0
        self.alpha = 1.

        self.G = nn.ModuleList()
        blk = nn.ModuleList()
        blk.append(Conv2d(self.depths[0], self.depths[0], 4, padding=3)) # to 4x4
        blk.append(Conv2d(self.depths[0], self.depths[0], 3, padding=1))
        self.G.append(blk)
        self.toRGB = nn.ModuleList()
        self.toRGB.append(Conv2d(self.depths[0], self.img_channels, 1, lrelu=False, pnorm=False)) # toRGB

        self.fromRGB = nn.ModuleList()
        self.fromRGB.append(Conv2d(self.img_channels, self.depths[0], 1)) # fromRGB
        self.D = nn.ModuleList()
        blk = nn.ModuleList()
        blk.append(MinibatchStddev())
        blk.append(Conv2d(self.depths[0]+1, self.depths[0], 3, padding=1))
        blk.append(Conv2d(self.depths[0], self.depths[0], 4, stride=4)) # to 1x1
        blk.append(Flatten())
        blk.append(Linear(self.depths[0], 1))
        self.D.append(blk)

        self.doubling = nn.Upsample(scale_factor=2)
        self.halving = nn.AvgPool2d(2, 2)
        self.set_optimizer()
        self.criterion = GANLoss(loss_type=args.loss_type, device=device)
        self.loss_type = args.loss_type
    
    def generate(self, z):
        hz = z
        for idx in range(len(self.G)):
            for net in self.G[idx]:
                hz = net(hz)
            if idx == len(self.G)-2:
                res = hz
        xf = self.toRGB[self.didx](hz)
        if self.alpha < 1.0:
            res = self.toRGB[self.didx-1](res)
            res = self.doubling(res)
            xf = (1-self.alpha)*res + self.alpha*xf
        return xf
    
    def discriminate(self, x):
        nD = len(self.D)
        hy = self.fromRGB[self.didx](x)
        if self.alpha < 1.0:
            res = self.halving(x)
            res = self.fromRGB[self.didx-1](res)
        for idx in range(nD):
            for net in self.D[-idx-1]:
                hy = net(hy)
            if idx == 0 and self.alpha < 1.0:
                hy = (1-self.alpha)*res + self.alpha*hy
        y = hy
        return y
    
    def train_step(self, z, x):
        self.D_opt.zero_grad()
        xf = self.generate(z)
        
        yr = self.discriminate(x)
        
        yf = self.discriminate(xf.detach())

        dloss_r = self.criterion(yr, True)
        dloss_f = self.criterion(yf, False)
        dloss = dloss_r + dloss_f
        if self.loss_type == 'wgan-gp':
            gp = 10.*self._gp(x, xf.detach())
            dloss = dloss + gp
        dloss.backward(retain_graph=True)
        self.D_opt.step()

        self.G_opt.zero_grad()
        yf = self.discriminate(xf)

        gloss = self.criterion(yf, True)
        gloss.backward()
        self.G_opt.step()

        training_info = {
            'Dloss': dloss.item(),
            'Dloss_r': dloss_r.item(),
            'Dloss_f': dloss_f.item(),
            'Gloss': gloss.item(),
            'gp': gp.item(),
        }
        return training_info

    def add_scale(self, increase_idx=True):
        if increase_idx:
            self.didx += 1
        blk = nn.ModuleList()
        blk.append(nn.Upsample(scale_factor=2))
        blk.append(
            Conv2d(self.depths[self.didx-1], self.depths[self.didx], 3, padding=1)
        )
        blk.append(
            Conv2d(self.depths[self.didx], self.depths[self.didx], 3, padding=1)
        )
        self.G.append(blk)
        self.toRGB.append(Conv2d(self.depths[self.didx], self.img_channels, 1, lrelu=False, pnorm=False)) # toRGB

        self.fromRGB.append(Conv2d(self.img_channels, self.depths[self.didx], 1)) # fromRGB
        blk = nn.ModuleList()
        blk.append(
            Conv2d(self.depths[self.didx], self.depths[self.didx], 3, padding=1)
        )
        blk.append(
            Conv2d(self.depths[self.didx], self.depths[self.didx-1], 3, padding=1)
        )
        blk.append(
            nn.AvgPool2d(2, stride=2)
        )
        self.D.append(blk)
        self.to(self.device)
        self.set_optimizer()
        self.set_alpha(0.)
        
    def set_optimizer(self):
        dparams = list(self.D.parameters()) + list(self.fromRGB.parameters())
        gparams = list(self.G.parameters()) + list(self.toRGB.parameters())
        self.D_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, dparams),betas=[0., 0.99], lr=self.args.lr)
        self.G_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, gparams),betas=[0., 0.99], lr=self.args.lr)

    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def _gp(self, x, xf):
        N, C, H, W = x.size()
        eps = torch.rand(N, 1, 1, 1)
        eps = eps.expand(-1, C, H, W).to(self.device)
        interpolates = eps*x + (1-eps)*xf
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        yi = self.discriminate(interpolates)
        yi = yi.sum()

        yi_grad = torch.autograd.grad(outputs=yi, inputs=interpolates,
                                        create_graph=True, retain_graph=True)

        yi_grad = yi_grad[0].view(N, -1)
        yi_grad = torch.norm(yi_grad, p=2, dim=1)
        gp = torch.pow(yi_grad-1., 2).sum()
        return gp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIM = (128, 128, 3)

def tensor2img(tensor):
    img = (np.transpose(tensor.detach().cpu().numpy(), [1,2,0])+1)/2.
    img = np.clip(img, 0, 1)
    return img

def train():
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(
        root="drive/MyDrive/celeb_data", transform=transform
    )
    data_loader = DataLoader(
        dataset=dataset, batch_size=128,
        shuffle=True, num_workers=0
    )

    gan = PGGAN(args).to(DEVICE)
    global_step = 0
    if args.retrain:
        model = torch.load(args.load_model)
        global_step = model['global_step']
        global_step -= 1
        gan.didx = model['didx']
        for idx in range(gan.didx):
            gan.add_scale(increase_idx=False)
        gan.set_alpha(model['alpha'])
        print('global step: {}, Resolution: {}'.format(global_step, 4*2**gan.didx))
        gan.G.load_state_dict(model['G'])
        gan.D.load_state_dict(model['D'])
        gan.G_opt.load_state_dict(model['G_opt'])
        gan.D_opt.load_state_dict(model['D_opt'])
        gan.toRGB.load_state_dict(model['toRGB'])
        gan.fromRGB.load_state_dict(model['fromRGB'])
        
    nrows = 8 # for samples
    nrows_r = int(np.sqrt(args.batch_size))
    fix_z = torch.randn([nrows**2, args.zdim, 1, 1]).to(DEVICE)
    epochs = 0
    next_scale_step = 0
    alpha = 1.
    for i in range(gan.didx+1):
        next_scale_step += sum(args.scale_update_schedule[i])
    # next_scale_step = sum(args.scale_update_schedule[gan.didx])
    next_alpha_step = 0

    while global_step < args.max_step:
        with tqdm(enumerate(data_loader), total=len(data_loader), ncols=70) as t:
            t.set_description('{}x{}, Step {}, a {}'.format(4*2**gan.didx,4*2**gan.didx,global_step, alpha))
            for idx, (images, _) in t:
                stride = 2**(5-gan.didx)
                x = images[..., ::stride, ::stride].to(DEVICE)
                z = torch.randn([x.size(0), args.zdim, 1, 1]).to(DEVICE)
                tinfo = gan.train_step(z, x)

                global_step += 1

                if global_step % args.log_step == 0:

                    gan.eval()
                    with torch.no_grad():
                        xf = gan.generate(fix_z)
                    xf = torch.cat([torch.cat([xf[nrows*j+i] for i in range(nrows)], dim=1) for j in range(nrows)], dim=2)
                    imgs = tensor2img(xf)
                    plt.imsave('{}/{:04d}k.jpg'.format(args.sample_dir, global_step//1000), imgs)
                    gan.train()

                if global_step % args.save_step == 0:
                    save_model(gan, global_step)
                    print("Step : ", global_step)

                if global_step == next_scale_step and stride > 1:
                    print('\nScale up\n')
                    gan.add_scale()
                    next_scale_step = global_step + sum(args.scale_update_schedule[gan.didx])
                    alpha_idx = 0
                    alpha = args.scale_update_alpha[gan.didx][alpha_idx]
                    gan.set_alpha(alpha)
                    next_alpha_step = global_step + args.scale_update_schedule[gan.didx][alpha_idx]
                elif global_step == next_alpha_step:
                    alpha_idx += 1
                    alpha = args.scale_update_alpha[gan.didx][alpha_idx]
                    gan.set_alpha(alpha)
                    next_alpha_step = global_step + args.scale_update_schedule[gan.didx][alpha_idx]
                

            epochs += 1
        
            
def save_model(model, global_step):
    infos = {
        'G': model.G.state_dict(),
        'D': model.D.state_dict(),
        'G_opt': model.G_opt.state_dict(),
        'D_opt': model.D_opt.state_dict(),
        'toRGB': model.toRGB.state_dict(),
        'fromRGB': model.fromRGB.state_dict(),
        'global_step': global_step,
        'didx': model.didx,
        'alpha': model.alpha,
    }
    torch.save(infos, '{}/{}-{:04d}k.pth.tar'.format(args.ckpt_dir, "GANwProgressive", global_step//1000))

class ConfigArgs:
    loss_type = 'wgan-gp'
    log_dir = 'logs'
    sample_dir = os.path.join(log_dir, 'samples')
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    data_dir = "drive/MyDrive/celeba"
    retrain = False
    load_model = None
    batch_size = 256
    lr = 0.001
    zdim = 256
    max_step = 1000000
    log_step = 1000
    save_step = 2000
    scale_update_schedule = [
        [5000],
        [15000, 15000, 15000, 15000], 
        [15000, 15000, 15000, 15000], 
        [1500*10, 15000, 15000, 15000], 
        [15000, 15000, 15000, 15000] 
    ]
    scale_update_alpha = [
        [1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.25, 0.5, 0.75, 1.0]
    ]

args = ConfigArgs

train()
