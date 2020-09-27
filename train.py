import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from PIL import Image

from net import DenseFuse_net
from args_fusion import args
import pytorch_msssim

is_cuda = torch.cuda.is_available()

class MyTrainDataset(Dataset):
    def __init__(self, img_path_file, custom_transform=transforms.Compose([transforms.ToTensor()])):
        f = open(img_path_file, 'r')
        img_list = f.read().splitlines()
        f.close()

        self.img_list = img_list
        self.transform = custom_transform
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index])

        if self.transform:
            img = self.transform(img)
        
        return img

    def __len__(self):

        return len(self.img_list)

def compute_loss(predicts, targets, weights, w_idx=0):

    targets = Variable(targets.data.clone(), requires_grad=False)
    
    mse = nn.MSELoss()
    ssim = pytorch_msssim.SSIM()

    loss_mse = mse(predicts, targets)
    loss_ssim = 1 - ssim(predicts, targets)

    loss = loss_mse + weights[w_idx] * loss_ssim

    return loss

if __name__ == '__main__':

    model = DenseFuse_net()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    img_path_file = args.dataset
    custom_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize((args.HEIGHT,args.WIDTH)),
                                           transforms.ToTensor()])
    trainloader = DataLoader(MyTrainDataset(img_path_file, custom_transform=custom_transform), batch_size=args.batch_size, shuffle=True, num_workers=1)

    if is_cuda:
        model.cuda()

    for ep in range(epoch, args.epochs):

        for i, inputs in enumerate(tqdm(trainloader)):

            if is_cuda:
                inputs = inputs.cuda()

            optimizer.zero_grad()
            
            en = model.encoder(inputs)
            predicts = model.decoder(en)

            loss = compute_loss(predicts, inputs, args.ssim_weight, w_idx=0)
            loss.backward()
            optimizer.step()

        if (ep + 1) % args.save_per_epoch == 0:
            # Save model
            torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                       }, args.save_model_dir + 'ckpt_{}.pt'.format(ep))

    print('Finished training')    